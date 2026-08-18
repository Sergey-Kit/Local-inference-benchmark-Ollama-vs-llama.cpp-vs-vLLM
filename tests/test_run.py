"""Tests for the orchestrator's config plumbing.

These cover the substitutions that would fail *silently* rather than loudly:
a wrong context total still starts a server and still produces numbers, they
are just numbers for a different experiment than the one described.
"""

import json
from pathlib import Path

import pytest

from bench.run import (
    Point,
    expand_scenarios,
    load_config,
    load_prompts,
    substitution_vars,
    _fmt,
)

CONFIG = Path("configs/experiment.yaml")
PROMPTS = Path("prompts/prompts.jsonl")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


class TestSubstitution:
    def test_ctx_total_is_per_slot_times_parallel(self, cfg):
        # llama-server divides -c across --parallel slots, so the config must
        # hand it the product. Passing ctx_per_slot would give each slot
        # ctx/parallel and break the long-prompt scenario without erroring.
        profile = {"parallel": 16, "ctx_per_slot": 512}
        assert substitution_vars(cfg, profile)["ctx_total"] == 8192

    def test_llamacpp_command_receives_the_total(self, cfg):
        vars_ = substitution_vars(cfg, cfg["server_profiles"]["interactive"])
        cmd = [_fmt(p, vars_) for p in cfg["runtimes"]["llamacpp"]["command"]]
        assert cmd[cmd.index("-c") + 1] == "8192"
        assert cmd[cmd.index("--parallel") + 1] == "16"

    def test_ollama_context_length_is_per_slot(self, cfg):
        # Ollama's OLLAMA_CONTEXT_LENGTH is per slot, unlike llama.cpp's -c.
        # Getting these two the same way round is the whole point of the test.
        vars_ = substitution_vars(cfg, cfg["server_profiles"]["interactive"])
        env = {k: _fmt(v, vars_) for k, v in cfg["runtimes"]["ollama"]["env"].items()}
        assert env["OLLAMA_CONTEXT_LENGTH"] == "512"
        assert env["OLLAMA_NUM_PARALLEL"] == "16"

    def test_vllm_max_model_len_is_per_slot(self, cfg):
        vars_ = substitution_vars(cfg, cfg["server_profiles"]["longctx"])
        cmd = [_fmt(p, vars_) for p in cfg["runtimes"]["vllm"]["command"]]
        assert cmd[cmd.index("--max-model-len") + 1] == "4096"
        assert cmd[cmd.index("--max-num-seqs") + 1] == "1"


class TestConfigInvariants:
    def test_every_runtime_is_on_a_distinct_port(self, cfg):
        ports = [rt["port"] for rt in cfg["runtimes"].values()]
        assert len(ports) == len(set(ports))

    def test_sampling_is_greedy_and_identical_for_all(self, cfg):
        assert cfg["sampling"]["temperature"] == 0.0
        assert cfg["sampling"]["top_p"] == 1.0
        assert cfg["sampling"]["max_tokens"] > 0

    def test_vllm_forces_fp16_and_no_prefix_cache(self, cfg):
        cmd = [str(p) for p in cfg["runtimes"]["vllm"]["command"]]
        # sm75 has no bfloat16; a bf16 default would fail or fall back silently.
        assert cmd[cmd.index("--dtype") + 1] == "float16"
        assert "--no-enable-prefix-caching" in cmd

    def test_every_scenario_profile_exists(self, cfg):
        for scenario in cfg["scenarios"]:
            assert scenario["profile"] in cfg["server_profiles"]

    def test_longctx_profile_fits_the_long_prompt(self, cfg):
        """The long prompt plus its output must fit one slot."""
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        sets = load_prompts(PROMPTS)
        needed = sets["long"][0].n_tokens + cfg["sampling"]["max_tokens"]
        assert cfg["server_profiles"]["longctx"]["ctx_per_slot"] >= needed

    def test_interactive_profile_fits_the_short_prompt(self, cfg):
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        sets = load_prompts(PROMPTS)
        needed = sets["short"][0].n_tokens + cfg["sampling"]["max_tokens"]
        assert cfg["server_profiles"]["interactive"]["ctx_per_slot"] >= needed


class TestMatrix:
    def test_expands_to_the_documented_matrix(self, cfg):
        points = expand_scenarios(cfg)
        # Compare on what identifies a point, not on the whole dataclass: the
        # per-scenario tuning that rides along is not part of its identity.
        ids = {(p.scenario, p.profile, p.prompt_set, p.concurrency) for p in points}
        assert ("concurrency_sweep", "interactive", "short", 16) in ids
        assert ("prompt_length", "longctx", "long", 1) in ids
        assert len(points) == 6

    def test_enough_prompts_for_every_run_plus_warmup(self, cfg):
        """No prompt is ever seen twice by one server.

        Each run takes its own window of the prompt list, and warm-up takes the
        window past all of them. If the list is shorter than the sum, windows
        wrap and a repeat becomes a prompt-cache hit -- which is not a small
        effect: replaying run 0's prompts dropped llama.cpp's TTFT from 188 ms
        to 24 ms, so the median of three runs would have described the cache
        rather than the runtime.
        """
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        from bench.run import requests_for
        sets = load_prompts(PROMPTS)
        m = cfg["measurement"]
        warmup = int(m["warmup_requests"])
        for scenario in cfg["scenarios"]:
            names = scenario["prompt_set"]
            names = [names] if isinstance(names, str) else names
            floor = {"requests_per_run_min": scenario["requests_per_run_min"]} \
                if "requests_per_run_min" in scenario else {}
            needed = max(requests_for(m, c, floor) for c in scenario["concurrency"]) * int(m["n_runs"])
            for name in names:
                assert len(sets[name]) >= needed + warmup, (
                    f"{name}: {len(sets[name])} prompts, need {needed + warmup}"
                )


class TestPrompts:
    def test_lengths_are_exact_and_uniform(self):
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        sets = load_prompts(PROMPTS)
        assert {p.n_tokens for p in sets["short"]} == {64}
        assert {p.n_tokens for p in sets["long"]} == {2560}

    def test_prompts_do_not_share_prefixes(self):
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        sets = load_prompts(PROMPTS)
        texts = [p.text for ps in sets.values() for p in ps]
        assert len({t[:400] for t in texts}) == len(texts)

    def test_thinking_mode_is_disabled(self):
        """Qwen3's thinking mode would make output length a hidden variable."""
        if not PROMPTS.exists():
            pytest.skip("prompts not built yet")
        rows = [json.loads(l) for l in PROMPTS.read_text().splitlines() if l.strip()]
        for row in rows:
            assert row["text"].rstrip().endswith("</think>")

    def test_rejects_mixed_length_sets(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps({"name": "a", "prompt_set": "short", "text": "x", "n_tokens": 64}) + "\n"
            + json.dumps({"name": "b", "prompt_set": "short", "text": "y", "n_tokens": 65}) + "\n"
        )
        with pytest.raises(SystemExit):
            load_prompts(bad)


class TestRequestCount:
    def test_scenario_may_lower_the_floor(self, cfg):
        """The long-prompt scenario opts out of the sweep's request budget.

        A 2560-token prefill costs ~6 s here against <1 s of generation, so the
        sweep's 16 requests would dominate the whole matrix's wall clock to
        measure a low-variance single-stream number.
        """
        from bench.run import requests_for
        m = cfg["measurement"]
        long_scenario = next(s for s in cfg["scenarios"] if s["name"] == "prompt_length")
        floor = {"requests_per_run_min": long_scenario["requests_per_run_min"]}
        assert requests_for(m, 1, floor) < requests_for(m, 1)
        assert requests_for(m, 1, floor) * int(m["n_runs"]) >= 20  # still enough samples

    def test_scales_with_concurrency(self, cfg):
        from bench.run import requests_for
        m = cfg["measurement"]
        # A synchronized single wave (n == concurrency) would never exercise the
        # scheduler's ability to backfill freed slots, which is the entire
        # mechanism the throughput figure is meant to reveal.
        for c in (1, 4, 8, 16):
            assert requests_for(m, c) >= max(16, 2 * c)

    def test_never_below_the_floor(self, cfg):
        from bench.run import requests_for
        assert requests_for(cfg["measurement"], 1) == 16


class TestNoProxy:
    def test_client_ignores_proxy_environment(self, monkeypatch):
        """The runtime is always on localhost and must never be proxied.

        This box sets http_proxy=http://127.0.0.1:12334 with no no_proxy, and
        httpx honours those variables by default -- which turns every request
        to the runtime into a 502 from the proxy.
        """
        import asyncio

        from bench.client import RuntimeClient

        monkeypatch.setenv("http_proxy", "http://127.0.0.1:12334")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:12334")
        monkeypatch.delenv("no_proxy", raising=False)

        async def check():
            async with RuntimeClient("t", "http://127.0.0.1:8080/v1", "m") as c:
                assert c.client.trust_env is False
                transport = c.client._mounts
                assert not transport, f"proxy mounts configured: {transport}"

        asyncio.run(check())


class TestRestartPolicy:
    def test_longctx_restarts_between_runs(self, cfg):
        """The long-prompt profile must not reuse a leaking server across runs.

        llama-server grows ~295 MiB of host RSS per 2560-token request here;
        3 runs x 16 requests on one server exceeds 5.5 GB of RAM and the kernel
        kills it part-way through the measurement.
        """
        assert cfg["server_profiles"]["longctx"].get("restart_between_runs") is True

    def test_interactive_restarts_too(self, cfg):
        """Short prompts leak less, but they still leak enough to matter.

        Reusing one server across the concurrency sweep, throughput fell run
        over run in lockstep with RSS -- 216 -> 166 -> 144 tok/s at concurrency
        16 while RSS climbed 2709 -> 3023 -> 3923 MiB. The median of three runs
        then describes a degrading process, and which value it lands on depends
        on run order.
        """
        assert cfg["server_profiles"]["interactive"].get("restart_between_runs") is True

    def test_every_profile_restarts(self, cfg):
        for name, profile in cfg["server_profiles"].items():
            assert profile.get("restart_between_runs") is True, name


class TestOllamaModelStore:
    def test_points_at_the_in_tree_store(self, cfg):
        """Ollama must read the F16 import, not ~/.ollama.

        Without OLLAMA_MODELS the server starts happily against the default
        store, fails to find the tag, and the obvious "fix" is `ollama pull` --
        which silently substitutes Ollama's own Q4_K_M for the F16 weights the
        other two runtimes are using.
        """
        from bench.run import REPO_ROOT, substitution_vars, _fmt
        raw = cfg["runtimes"]["ollama"]["env"]["OLLAMA_MODELS"]
        resolved = _fmt(raw, substitution_vars(cfg, cfg["server_profiles"]["interactive"]))
        # Absolute: Ollama does not honour a relative store path, and the
        # failure mode is a server that starts fine and then 404s on the model.
        assert resolved.startswith("/")
        assert resolved == f"{REPO_ROOT}/vendor/ollama/models"

    def test_the_tag_exists_in_that_store(self, cfg):
        import json
        from pathlib import Path
        from bench.run import substitution_vars, _fmt
        store = Path(_fmt(cfg["runtimes"]["ollama"]["env"]["OLLAMA_MODELS"],
                          substitution_vars(cfg, cfg["server_profiles"]["interactive"])))
        if not store.exists():
            pytest.skip("ollama store not populated yet")
        tag = cfg["model"]["ollama_tag"]
        name, _, version = tag.partition(":")
        found = list((store / "manifests").rglob(f"{name}/{version or 'latest'}"))
        assert found, f"{tag} not imported into {store}"
        manifest = json.loads(found[0].read_text())
        layers = [l for l in manifest["layers"]
                  if l["mediaType"] == "application/vnd.ollama.image.model"]
        assert len(layers) == 1


class TestServerEnvironment:
    def test_env_block_is_substituted_and_complete(self, cfg):
        """The `env:` block must survive substitution intact.

        It was for a while built and then dropped on the floor -- never passed
        to Popen -- so Ollama would have been measured at its default
        parallelism and context instead of the configured 16 and 512. That is
        invisible in the output: the server starts, answers every request, and
        reports numbers for a different experiment.
        """
        from pathlib import Path

        from bench.run import ServerManager

        mgr = ServerManager(cfg, "ollama", Path("."))
        env = mgr.env_overrides(cfg["server_profiles"]["interactive"])
        assert env["OLLAMA_NUM_PARALLEL"] == "16"
        assert env["OLLAMA_CONTEXT_LENGTH"] == "512"
        assert env["OLLAMA_MODELS"].startswith("/")
        assert "{" not in "".join(env.values()), "unsubstituted placeholder left in env"

    def test_launch_passes_env_to_the_subprocess(self, cfg, monkeypatch):
        from pathlib import Path

        import bench.run as run_mod

        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                captured["cmd"] = cmd
                captured["env"] = kw.get("env")
                self.pid = 1

            def poll(self):
                return None

        monkeypatch.setattr(run_mod.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(run_mod, "port_is_open", lambda *a, **k: False)
        mgr = run_mod.ServerManager(cfg, "ollama", Path("/tmp"))
        mgr.launch("interactive", cfg["server_profiles"]["interactive"])
        assert captured["env"] is not None, "Popen was called without env"
        assert captured["env"]["OLLAMA_NUM_PARALLEL"] == "16"
        mgr.log_file.close()
