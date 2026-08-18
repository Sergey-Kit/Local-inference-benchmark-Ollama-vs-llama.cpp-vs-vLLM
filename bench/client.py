"""One streaming OpenAI-compatible client, shared by all three runtimes.

This is the whole point of the architecture in SPEC section 3: because Ollama,
llama.cpp and vLLM all speak /v1, the measuring code exists exactly once, so a
difference in the numbers is a difference in the runtime and not in the client.

Two deliberate departures from the SPEC, both in the name of comparability:

* We call /v1/completions, not /v1/chat/completions. The chat template is
  applied inside each runtime -- vLLM from the HF repo, llama.cpp and Ollama
  from the template baked into the GGUF -- and any divergence changes the
  prompt token count without saying so, which would quietly invalidate every
  prefill/TTFT comparison. The template is applied once, ahead of time, by
  prompts/build_prompts.py, and the rendered string is sent verbatim.

* Token counts are cross-checked. `usage` from the runtime is preferred, but it
  is verified against our own tokenizer; a disagreement is recorded rather than
  averaged away.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import httpx

from bench.metrics import RequestRecord


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = True


@dataclass(frozen=True)
class Prompt:
    name: str
    prompt_set: str
    text: str
    n_tokens: int


class TokenCountMismatch(RuntimeError):
    """The runtime's usage and our tokenizer disagree beyond tolerance."""


class RuntimeClient:
    """Talks to one already-running runtime."""

    def __init__(
        self,
        name: str,
        base_url: str,
        served_model: str,
        *,
        health_path: str = "/health",
        supports_ignore_eos: bool = True,
        timeout_s: float = 600.0,
        count_tokens: Callable[[str], int] | None = None,
        token_mismatch_tolerance: float = 0.01,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.health_path = health_path
        self.supports_ignore_eos = supports_ignore_eos
        self.timeout_s = timeout_s
        self.count_tokens = count_tokens
        self.token_mismatch_tolerance = token_mismatch_tolerance
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "RuntimeClient":
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
        # trust_env=False: httpx honours HTTP_PROXY/HTTPS_PROXY by default, and
        # this machine has http_proxy=http://127.0.0.1:12334 set with no
        # no_proxy. Requests to the runtime on 127.0.0.1 would be routed through
        # that proxy, which answers 502 -- so every measurement would fail, or
        # worse, succeed while timing the proxy hop. The server is always local;
        # it must never be reached through a proxy.
        self._client = httpx.AsyncClient(
            timeout=self.timeout_s, limits=limits, trust_env=False
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("RuntimeClient must be used as an async context manager")
        return self._client

    async def wait_ready(self, timeout_s: float = 600.0, poll_s: float = 0.25) -> float:
        """Block until the server answers a real generation request.

        Returns seconds waited. This is the cold-start metric from SPEC section
        2, and it deliberately waits for a *completed generation* rather than a
        200 on /health: several runtimes bind the port and report healthy while
        the model is still loading, which would make cold start look far better
        than it is.
        """
        started = time.monotonic()
        deadline = started + timeout_s
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                resp = await self.client.post(
                    f"{self.base_url}/completions",
                    json={
                        "model": self.served_model,
                        "prompt": "ping",
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    return time.monotonic() - started
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(poll_s)
        raise TimeoutError(
            f"{self.name} did not become ready within {timeout_s}s; last error: {last_error}"
        )

    # -- measurement -------------------------------------------------------

    def _payload(self, prompt: str, sampling: Sampling) -> dict:
        payload: dict = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": True,
            # Without this, a streamed response carries no usage block at all
            # and we would be left counting words -- which SPEC section 3
            # explicitly warns against.
            "stream_options": {"include_usage": True},
        }
        if sampling.ignore_eos and self.supports_ignore_eos:
            # Makes every runtime emit exactly max_tokens, so TPOT and
            # throughput are not skewed by differing stop behaviour.
            payload["ignore_eos"] = True
        return payload

    async def one_request(
        self,
        prompt: Prompt,
        sampling: Sampling,
        *,
        scenario: str,
        concurrency: int,
        run_index: int,
    ) -> RequestRecord:
        record_kw = dict(
            runtime=self.name,
            scenario=scenario,
            prompt_set=prompt.prompt_set,
            concurrency=concurrency,
            run_index=run_index,
        )
        started = time.monotonic()
        ttft_s: float | None = None
        text_parts: list[str] = []
        usage: dict | None = None
        finish_reason: str | None = None
        n_chunks = 0
        saw_done = False
        stream_error: str | None = None

        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/completions", json=self._payload(prompt.text, sampling)
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {body}", request=resp.request, response=resp
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    n_chunks += 1
                    if chunk.get("error"):
                        # llama-server reports some failures *inside* the SSE
                        # stream with HTTP 200 already sent. Ignoring these
                        # chunks -- as any client that only looks at
                        # choices[].text does -- turns a half-finished
                        # generation into a plausible-looking success.
                        err = chunk["error"]
                        stream_error = str(err.get("message", err))[:200]
                        break
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        piece = choice.get("text") or ""
                        if not piece:
                            continue
                        if ttft_s is None:
                            # First *non-empty* chunk: some runtimes emit a
                            # priming chunk with an empty string.
                            ttft_s = time.monotonic() - started
                        text_parts.append(piece)
        except Exception as exc:  # noqa: BLE001 - a failed request is a datum
            return RequestRecord(
                prompt_tokens=prompt.n_tokens,
                output_tokens=0,
                ttft_s=ttft_s,
                total_s=time.monotonic() - started,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
                finish_reason=finish_reason,
                n_chunks=n_chunks,
                **record_kw,
            )

        total_s = time.monotonic() - started
        completion = "".join(text_parts)
        output_tokens, source = self._resolve_output_tokens(usage, completion)
        prompt_tokens = int(usage["prompt_tokens"]) if usage and "prompt_tokens" in usage else prompt.n_tokens

        # A trustworthy request is one that generated tokens AND was seen
        # through to a proper end. A stream that stops early still yields a
        # respectable-looking token count and a respectable-looking TTFT; only
        # the terminator says whether the generation actually finished.
        failure: str | None = None
        if stream_error is not None:
            failure = f"stream error: {stream_error}"
        elif output_tokens <= 0:
            failure = "no tokens generated"
        elif not saw_done and finish_reason is None:
            failure = f"stream ended after {n_chunks} chunks without finish_reason or [DONE]"

        return RequestRecord(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft_s,
            total_s=total_s,
            ok=failure is None,
            error=failure,
            token_count_source=source,
            finish_reason=finish_reason,
            n_chunks=n_chunks,
            **record_kw,
        )

    def _resolve_output_tokens(self, usage: dict | None, completion: str) -> tuple[int, str]:
        """Prefer the runtime's usage, but never trust it blindly."""
        ours = self.count_tokens(completion) if self.count_tokens and completion else None
        theirs = None
        if usage and usage.get("completion_tokens") is not None:
            theirs = int(usage["completion_tokens"])

        if theirs is None:
            if ours is None:
                return 0, "unavailable"
            return ours, "tokenizer"
        if ours is None or theirs == 0:
            return theirs, "usage"

        drift = abs(theirs - ours) / max(theirs, 1)
        if drift > self.token_mismatch_tolerance:
            # Recorded, not silently reconciled: a systematic disagreement
            # means the runtimes are not tokenizing the same way, which is a
            # finding about the comparison itself.
            return theirs, f"usage(tokenizer_drift={drift:.1%})"
        return theirs, "usage"

    # -- load generation ---------------------------------------------------

    async def run_batch(
        self,
        prompts: Sequence[Prompt],
        sampling: Sampling,
        *,
        scenario: str,
        concurrency: int,
        n_requests: int,
        run_index: int,
    ) -> tuple[list[RequestRecord], float]:
        """Issue `n_requests` with at most `concurrency` in flight.

        A semaphore keeps exactly `concurrency` requests outstanding, which is
        what makes the throughput-vs-concurrency curve mean something: the
        server is asked to hold N streams open, not to absorb a burst of N.
        """
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int) -> RequestRecord:
            async with sem:
                return await self.one_request(
                    prompts[i % len(prompts)],
                    sampling,
                    scenario=scenario,
                    concurrency=concurrency,
                    run_index=run_index,
                )

        started = time.monotonic()
        records = await asyncio.gather(*(one(i) for i in range(n_requests)))
        return list(records), time.monotonic() - started

    async def warmup(self, prompt: Prompt, sampling: Sampling, n: int = 8) -> None:
        """Discarded traffic: fills caches and opens the connection pool."""
        await self.run_batch(
            [prompt], sampling, scenario="warmup", concurrency=min(n, 4),
            n_requests=n, run_index=-1,
        )
