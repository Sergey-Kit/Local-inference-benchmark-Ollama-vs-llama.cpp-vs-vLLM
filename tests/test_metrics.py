import math

import pytest

from bench.metrics import (
    MetricsError,
    RequestRecord,
    median_across_runs,
    percentile,
    summarize_run,
    tpot_ms,
    ttft_ms,
)


def rec(**kw):
    base = dict(
        runtime="llamacpp",
        scenario="concurrency_sweep",
        prompt_set="short",
        concurrency=1,
        run_index=0,
        prompt_tokens=64,
        output_tokens=128,
        ttft_s=0.1,
        total_s=1.37,
    )
    base.update(kw)
    return RequestRecord(**base)


class TestTtftTpot:
    def test_ttft_converts_to_ms(self):
        assert ttft_ms(rec(ttft_s=0.25)) == pytest.approx(250.0)

    def test_ttft_none_when_no_first_token(self):
        assert ttft_ms(rec(ttft_s=None)) is None

    def test_tpot_matches_spec_formula(self):
        # (1.37 - 0.10) / (128 - 1) = 0.01 s = 10 ms
        assert tpot_ms(rec()) == pytest.approx(10.0)

    def test_tpot_none_for_single_token(self):
        # One token means there is no inter-token interval to measure.
        assert tpot_ms(rec(output_tokens=1)) is None

    def test_tpot_rejects_backwards_clock(self):
        with pytest.raises(MetricsError):
            tpot_ms(rec(ttft_s=2.0, total_s=1.0))


class TestPercentile:
    def test_matches_numpy_linear_interpolation(self):
        values = [1, 2, 3, 4]
        assert percentile(values, 50.0) == pytest.approx(2.5)
        assert percentile(values, 95.0) == pytest.approx(3.85)

    def test_single_value(self):
        assert percentile([7.5], 95.0) == pytest.approx(7.5)

    def test_is_order_independent(self):
        assert percentile([9, 1, 5], 50.0) == percentile([1, 5, 9], 50.0)

    def test_rejects_empty_and_out_of_range(self):
        with pytest.raises(MetricsError):
            percentile([], 50.0)
        with pytest.raises(MetricsError):
            percentile([1.0], 101.0)


class TestSummarizeRun:
    def test_throughput_and_percentiles(self):
        records = [rec(ttft_s=0.1), rec(ttft_s=0.2), rec(ttft_s=0.3)]
        s = summarize_run(records, wall_clock_s=4.0)
        assert s.total_output_tokens == 384
        assert s.throughput_tok_s == pytest.approx(96.0)
        assert s.ttft_ms_p50 == pytest.approx(200.0)
        assert s.n_failed == 0
        assert s.notes == []

    def test_failed_requests_still_consume_wall_clock(self):
        # A runtime that drops requests under load must not look faster for it:
        # the failure contributes no tokens but the clock still runs.
        records = [rec(), rec(ok=False, output_tokens=0, error="timeout")]
        s = summarize_run(records, wall_clock_s=2.0)
        assert s.n_failed == 1
        assert s.total_output_tokens == 128
        assert s.throughput_tok_s == pytest.approx(64.0)
        assert "1/2 requests failed" in s.notes

    def test_notes_flag_non_usage_token_counts(self):
        s = summarize_run([rec(token_count_source="tokenizer")], wall_clock_s=1.0)
        assert any("tokenizer" in n for n in s.notes)

    def test_rejects_empty_or_zero_clock(self):
        with pytest.raises(MetricsError):
            summarize_run([], wall_clock_s=1.0)
        with pytest.raises(MetricsError):
            summarize_run([rec()], wall_clock_s=0.0)


class TestMedianAcrossRuns:
    def test_takes_median_not_mean(self):
        summaries = [
            summarize_run([rec(run_index=i, ttft_s=t)], wall_clock_s=w)
            for i, (t, w) in enumerate([(0.1, 1.0), (0.2, 2.0), (0.9, 10.0)])
        ]
        agg = median_across_runs(summaries)
        assert agg["n_runs"] == 3
        assert agg["ttft_ms_p50"] == pytest.approx(200.0)
        # medians of 128.0, 64.0, 12.8 -> 64.0, not the mean 68.3
        assert agg["throughput_tok_s"] == pytest.approx(64.0)

    def test_refuses_to_mix_measurement_points(self):
        a = summarize_run([rec(runtime="llamacpp")], wall_clock_s=1.0)
        b = summarize_run([rec(runtime="vllm")], wall_clock_s=1.0)
        with pytest.raises(MetricsError):
            median_across_runs([a, b])

    def test_missing_optional_metrics_stay_none(self):
        s = summarize_run([rec()], wall_clock_s=1.0)
        agg = median_across_runs([s])
        assert agg["peak_vram_mib"] is None
        assert agg["cold_start_s"] is None
