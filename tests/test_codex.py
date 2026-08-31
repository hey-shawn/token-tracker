import json
import os
from datetime import UTC, datetime
from pathlib import Path

from token_tracker.adapters import codex


def _write_session(tmp_path: Path, events: list[dict], name: str = "session.jsonl") -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return p


def _token_count_event(
    rate_limits: dict,
    context_window: int | None = 258400,
    timestamp: str = "2026-06-04T20:00:00.000Z",
) -> dict:
    info = {"total_token_usage": {"input_tokens": 1, "output_tokens": 1}}
    if context_window is not None:
        info["model_context_window"] = context_window
    return {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": info,
            "rate_limits": {"limit_id": "codex", **rate_limits},
        },
    }


def test_virtual_model_rewritten_to_real_model():
    # codex-auto-review 是 Codex stop-time auto-review gate 的虚拟 model name，
    # 应改写为背后真实模型（gpt-5.5），避免在 Model Trend 等报表里独占一行
    assert codex._rewrite_virtual_model("codex-auto-review") == "gpt-5.5"
    assert codex._rewrite_virtual_model("gpt-5.5") == "gpt-5.5"
    assert codex._rewrite_virtual_model("gpt-5-codex") == "gpt-5-codex"
    assert codex._rewrite_virtual_model("unknown") == "unknown"


def test_session_end_recorded_from_last_event(tmp_path):
    # codex 单条 entry 记录会话最后事件时间作 session_end，供 aggregate_sessions 算真实跨度
    events = [
        {"timestamp": "2026-06-22T10:00:00.000Z", "type": "session_meta",
         "payload": {"id": "s1", "timestamp": "2026-06-22T10:00:00.000Z", "cwd": "/tmp/proj"}},
        {"timestamp": "2026-06-22T10:03:00.000Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 50}}}},
        {"timestamp": "2026-06-22T10:12:30.000Z", "type": "event_msg",
         "payload": {"type": "task_complete", "duration_ms": 5000}},
    ]
    path = _write_session(tmp_path, events)
    entries: list = []
    codex._parse_jsonl(path, {}, entries, set(), None)
    assert len(entries) == 1
    assert entries[0].timestamp.isoformat() == "2026-06-22T10:00:00+00:00"
    assert entries[0].session_end.isoformat() == "2026-06-22T10:12:30+00:00"


def test_codex_single_entry_yields_real_duration():
    # 回归：codex 每会话仅 1 条 entry，靠 session_end 让 aggregate_sessions 算出真实跨度（旧版恒为 0）
    from token_tracker.adapters.types import UsageEntry
    from token_tracker.analyzer.aggregator import aggregate_sessions
    e = UsageEntry(
        timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=UTC),
        session_id="s1", message_id="s1", request_id="", model="gpt-5.5",
        input_tokens=100, output_tokens=50, cache_creation_tokens=0, cache_read_tokens=0,
        cost_usd=None, project="p", agent_id="codex",
        session_end=datetime(2026, 6, 22, 10, 12, 30, tzinfo=UTC),
    )
    s = aggregate_sessions([e])[0]
    assert s.duration_minutes == 12.5
    assert s.end_time == datetime(2026, 6, 22, 10, 12, 30, tzinfo=UTC)


def test_free_plan_7d_bucket_routed_correctly(tmp_path):
    # Free plan: primary is the 7-day window (10080 min), secondary is null.
    # Old code put primary into the 5h slot, leaving 7d empty.
    rl = {
        "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
        "plan_type": "free",
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.seven_day_pct == 42.0
    assert result.five_hour_pct is None
    assert result.plan_type == "free"
    assert result.context_window == 258400


def test_paid_plan_both_buckets_routed(tmp_path):
    rl = {
        "primary": {"used_percent": 12.0, "window_minutes": 300, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 60.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "plan_type": "pro",
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.five_hour_pct == 12.0
    assert result.seven_day_pct == 60.0
    assert result.plan_type == "pro"


def test_swapped_window_order_still_routed_by_window_minutes(tmp_path):
    # Defensive: if OpenAI ever swaps primary/secondary order, bucket assignment
    # must follow window_minutes, not positional convention.
    rl = {
        "primary": {"used_percent": 55.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 8.0, "window_minutes": 300, "resets_at": 9_999_999_999},
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.five_hour_pct == 8.0
    assert result.seven_day_pct == 55.0


def test_expired_reset_zeros_out_pct(tmp_path):
    rl = {
        "primary": {"used_percent": 99.0, "window_minutes": 10080, "resets_at": 1},
        "secondary": None,
    }
    path = _write_session(tmp_path, [_token_count_event(rl)])
    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.seven_day_pct == 0.0


def test_spark_pool_does_not_replace_standard_weekly_limit(tmp_path):
    standard = {
        "primary": {"used_percent": 73.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    spark = {
        "limit_id": "codex_bengalfox",
        "limit_name": "GPT-5.3-Codex-Spark",
        "primary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    path = _write_session(
        tmp_path,
        [
            _token_count_event(standard, timestamp="2026-06-04T20:00:00.000Z"),
            _token_count_event(spark, timestamp="2026-06-04T21:00:00.000Z"),
        ],
    )

    result = codex._extract_rate_limits(path, models={})

    assert result is not None
    assert result.seven_day_pct == 73.0


def test_spark_only_session_has_no_standard_weekly_limit(tmp_path):
    spark = {
        "limit_id": "codex_bengalfox",
        "limit_name": "GPT-5.3-Codex-Spark",
        "primary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    path = _write_session(tmp_path, [_token_count_event(spark)])

    assert codex._extract_rate_limits(path, models={}) is None


def test_reasoning_tokens_not_double_counted(tmp_path):
    # reasoning_output_tokens 是 output_tokens 的子集拆分（实测 total_tokens == input + output），
    # 旧版 output+reasoning 重复计，token 统计与 cost 都偏高
    events = [
        {"timestamp": "2026-06-22T10:00:00.000Z", "type": "session_meta",
         "payload": {"id": "s1", "timestamp": "2026-06-22T10:00:00.000Z", "cwd": "/tmp/proj"}},
        {"timestamp": "2026-06-22T10:03:00.000Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {"total_token_usage": {
             "input_tokens": 18406, "cached_input_tokens": 9216,
             "output_tokens": 1054, "reasoning_output_tokens": 803, "total_tokens": 19460}}}},
    ]
    path = _write_session(tmp_path, events)
    entries: list = []
    codex._parse_jsonl(path, {}, entries, set(), None)
    assert len(entries) == 1
    assert entries[0].input_tokens == 18406 - 9216
    assert entries[0].output_tokens == 1054  # 不含 reasoning 803
    assert entries[0].cache_read_tokens == 9216


def test_pricing_segments_preserve_each_request_and_skip_duplicate_snapshots(tmp_path):
    events = [
        {"timestamp": "2026-08-22T00:00:00.000Z", "type": "session_meta",
         "payload": {"id": "s1", "timestamp": "2026-08-22T00:00:00.000Z", "cwd": "/tmp/proj"}},
        {"timestamp": "2026-08-22T01:00:00.000Z", "type": "event_msg", "payload": {
            "type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10},
                "last_token_usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10},
            },
        }},
        {"timestamp": "2026-08-22T01:00:01.000Z", "type": "event_msg", "payload": {
            "type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10},
                "last_token_usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10},
            },
        }},
        {"timestamp": "2026-08-22T06:00:00.000Z", "type": "event_msg", "payload": {
            "type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 250, "cached_input_tokens": 70, "output_tokens": 30},
                "last_token_usage": {"input_tokens": 150, "cached_input_tokens": 50, "output_tokens": 20},
            },
        }},
    ]
    path = _write_session(tmp_path, events)
    entries: list = []
    codex._parse_jsonl(path, {"s1": "deepseek-v4-flash"}, entries, set(), None)

    assert len(entries) == 1
    entry = entries[0]
    assert (entry.input_tokens, entry.cache_read_tokens, entry.output_tokens) == (180, 70, 30)
    assert len(entry.pricing_segments) == 2
    assert (
        entry.pricing_segments[0].input_tokens,
        entry.pricing_segments[0].cache_read_tokens,
        entry.pricing_segments[0].output_tokens,
    ) == (80, 20, 10)
    assert entry.pricing_segments[1].timestamp == datetime(2026, 8, 22, 6, tzinfo=UTC)


def test_session_snapshot_collects_statusline_data_in_one_scan(tmp_path, monkeypatch):
    rl = {
        "primary": {"used_percent": 12.0, "window_minutes": 300, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 60.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
    }
    path = _write_session(tmp_path, [
        _meta_event("deepseek", session_id="snapshot-s1"),
        {"timestamp": "2026-06-04T19:30:00.000Z", "type": "turn_context",
         "payload": {"model": "deepseek-v4-flash", "effort": "high"}},
        _token_count_event(rl, timestamp="2026-06-04T20:00:00.000Z"),
    ])
    original = codex.iter_jsonl_dicts
    scans = 0

    def counted(path_arg):
        nonlocal scans
        scans += 1
        yield from original(path_arg)

    monkeypatch.setattr(codex, "iter_jsonl_dicts", counted)
    snapshot = codex.load_session_snapshot(path)

    assert scans == 1
    assert snapshot.session_id == "snapshot-s1"
    assert snapshot.provider == "deepseek"
    assert snapshot.model == "deepseek-v4-flash"
    assert snapshot.effort == "high"
    assert snapshot.rate_limits is not None and snapshot.rate_limits.five_hour_pct == 12.0
    assert snapshot.usage_entry is not None and snapshot.usage_entry.model == "deepseek-v4-flash"


def test_load_rate_limits_uses_newest_standard_event_across_sessions(tmp_path, monkeypatch):
    newer_standard = {
        "primary": {"used_percent": 73.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    older_standard = {
        "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    spark = {
        "limit_id": "codex_bengalfox",
        "limit_name": "GPT-5.3-Codex-Spark",
        "primary": {"used_percent": 0.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    standard_path = _write_session(
        tmp_path,
        [_token_count_event(newer_standard, timestamp="2026-06-04T21:00:00.000Z")],
        "standard.jsonl",
    )
    spark_path = _write_session(
        tmp_path,
        [
            _token_count_event(older_standard, timestamp="2026-06-04T20:00:00.000Z"),
            _token_count_event(spark, timestamp="2026-06-04T22:00:00.000Z"),
        ],
        "spark.jsonl",
    )
    os.utime(standard_path, (100, 100))
    os.utime(spark_path, (200, 200))
    monkeypatch.setattr(codex, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(codex, "_load_thread_models", lambda: {})

    result = codex.load_rate_limits()

    assert result is not None
    assert result.seven_day_pct == 73.0


def _meta_event(provider: str, session_id: str = "s1") -> dict:
    return {
        "timestamp": "2026-06-04T19:00:00.000Z",
        "type": "session_meta",
        "payload": {"id": session_id, "timestamp": "2026-06-04T19:00:00.000Z",
                    "cwd": "/tmp/proj", "model_provider": provider},
    }


def test_load_rate_limits_provider_filter_skips_other_accounts(tmp_path, monkeypatch):
    # 同一 CODEX_HOME 混跑 openai 主账号 + deepseek profile：按 provider 过滤后
    # 不得把 openai 会话的配额显示在 deepseek 会话上（反之亦然）
    openai_rl = {
        "primary": {"used_percent": 73.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    deepseek_rl = {
        "primary": {"used_percent": 5.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    openai_path = _write_session(
        tmp_path,
        [_meta_event("openai"), _token_count_event(openai_rl, timestamp="2026-06-04T21:00:00.000Z")],
        "openai.jsonl",
    )
    ds_path = _write_session(
        tmp_path,
        [_meta_event("deepseek"), _token_count_event(deepseek_rl, timestamp="2026-06-04T22:00:00.000Z")],
        "deepseek.jsonl",
    )
    os.utime(openai_path, (100, 100))
    os.utime(ds_path, (200, 200))  # deepseek 文件更新：无过滤时会错误命中它
    monkeypatch.setattr(codex, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(codex, "_load_thread_models", lambda: {})

    assert codex.load_rate_limits().seven_day_pct == 5.0  # 不过滤：最新文件（保持旧行为）
    assert codex.load_rate_limits(provider="openai").seven_day_pct == 73.0
    assert codex.load_rate_limits(provider="deepseek").seven_day_pct == 5.0


def test_load_rate_limits_provider_filter_no_match_returns_none(tmp_path, monkeypatch):
    # 第三方 provider（deepseek）通常没有 codex 标准配额事件：过滤后应为 None，
    # 状态栏整段 Limit 不显示，而不是错拿 openai 账号的配额
    openai_rl = {
        "primary": {"used_percent": 73.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
        "secondary": None,
    }
    _write_session(
        tmp_path,
        [_meta_event("openai"), _token_count_event(openai_rl)],
    )
    monkeypatch.setattr(codex, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(codex, "_load_thread_models", lambda: {})

    assert codex.load_rate_limits(provider="deepseek") is None


def test_load_session_rate_limits_reads_single_transcript(tmp_path):
    rl = {
        "primary": {"used_percent": 12.0, "window_minutes": 300, "resets_at": 9_999_999_999},
        "secondary": {"used_percent": 60.0, "window_minutes": 10080, "resets_at": 9_999_999_999},
    }
    path = _write_session(tmp_path, [_meta_event("openai"), _token_count_event(rl)])

    result = codex.load_session_rate_limits(path)

    assert result is not None
    assert result.five_hour_pct == 12.0
    assert result.seven_day_pct == 60.0


def test_session_provider_reads_session_meta(tmp_path):
    path = _write_session(tmp_path, [_meta_event("deepseek")])
    assert codex.session_provider(path) == "deepseek"
    empty = _write_session(tmp_path, [], "empty.jsonl")
    assert codex.session_provider(empty) == ""


def test_latest_session_provider_picks_newest_with_meta(tmp_path, monkeypatch):
    older = _write_session(tmp_path, [_meta_event("openai")], "older.jsonl")
    newer = _write_session(tmp_path, [_meta_event("deepseek")], "newer.jsonl")
    no_meta = _write_session(tmp_path, [], "nometa.jsonl")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    os.utime(no_meta, (300, 300))  # 最新但无 session_meta → 跳过取次新的
    monkeypatch.setattr(codex, "SESSIONS_DIR", str(tmp_path))

    assert codex.latest_session_provider() == "deepseek"
    monkeypatch.setattr(codex, "SESSIONS_DIR", str(tmp_path / "nonexistent"))
    assert codex.latest_session_provider() == ""
