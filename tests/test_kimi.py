"""Kimi Code adapter：wire.jsonl 的 usage.record → UsageEntry。

目录布局照 tests/test_sidebar.py 的 _write_kimi_session：
`<sessions>/<wd_*>/<session_*>/agents/main/wire.jsonl` + 会话目录下 state.json（cwd/workDir）。
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from token_tracker.adapters import kimi


def _ms(seconds_ago: float = 60) -> int:
    return int((datetime.now(UTC) - timedelta(seconds=seconds_ago)).timestamp() * 1000)


def _usage_record(seconds_ago: float = 60, model: str = "kimi-code/k3", **usage: int) -> dict:
    return {"type": "usage.record", "model": model, "usage": usage, "time": _ms(seconds_ago)}


def _write_session(sessions_dir: Path, sid: str, rows: list[dict], work_dir: str = "") -> Path:
    session_dir = sessions_dir / "wd_myproj_abc123def456" / sid
    wire = session_dir / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True, exist_ok=True)
    with open(wire, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (session_dir / "state.json").write_text(json.dumps({"workDir": work_dir}), encoding="utf-8")
    return wire


def _load(sessions_dir: Path, hours_back: int = 0, monkeypatch=None) -> list:
    if monkeypatch is not None:
        monkeypatch.setattr(kimi, "SESSIONS_DIR", str(sessions_dir))
    return kimi.load_entries(hours_back)


def test_detect_uses_install_root_without_requiring_sessions(tmp_path, monkeypatch):
    kimi_dir = tmp_path / ".kimi-code"
    monkeypatch.setattr(kimi, "KIMI_DIR", str(kimi_dir))
    assert kimi.detect() is None
    kimi_dir.mkdir()
    info = kimi.detect()
    assert info is not None and info.id == "kimi" and info.name == "Kimi Code"


def test_usage_records_become_entries(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    proj = tmp_path / "myproj"
    _write_session(sessions_dir, "session_s1", [
        {"type": "turn.prompt", "input": [], "time": _ms(130)},  # 非 usage 事件忽略
        _usage_record(120, inputOther=4461, output=191, inputCacheRead=19200, inputCacheCreation=0),
        _usage_record(60, inputOther=530, output=175, inputCacheRead=23552, inputCacheCreation=10),
    ], work_dir=str(proj))

    entries = _load(sessions_dir, monkeypatch=monkeypatch)
    assert len(entries) == 2
    first, second = entries  # 按时间正序
    assert first.agent_id == "kimi"
    assert first.session_id == "session_s1"
    assert first.model == "kimi-code/k3"
    assert first.project == "myproj"
    assert first.input_tokens == 4461
    assert first.output_tokens == 191
    assert first.cache_read_tokens == 19200
    assert first.cache_creation_tokens == 0
    assert first.cost_usd is None  # 走 calculate_cost 定价表
    assert first.dedup_key != second.dedup_key
    assert second.cache_creation_tokens == 10


def test_usage_records_use_v2_cwd_for_project(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    proj = tmp_path / "v2-project"
    wire = _write_session(
        sessions_dir,
        "session_v2",
        [_usage_record(60, inputOther=1)],
    )
    (wire.parents[2] / "state.json").write_text(json.dumps({"cwd": str(proj)}), encoding="utf-8")

    entries = _load(sessions_dir, monkeypatch=monkeypatch)

    assert entries[0].project == "v2-project"


def test_zero_usage_records_skipped(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "session_s1", [
        _usage_record(60, inputOther=0, output=0, inputCacheRead=0, inputCacheCreation=0),
        _usage_record(50, inputOther=0, output=0),  # 字段缺失同样按 0 丢弃
        _usage_record(40, inputOther=1, output=0),
    ])
    entries = _load(sessions_dir, monkeypatch=monkeypatch)
    assert len(entries) == 1
    assert entries[0].input_tokens == 1


def test_hours_back_cutoff_filters_old_records(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "session_s1", [
        _usage_record(3600 * 10, inputOther=100, output=1),  # 10 小时前
        _usage_record(60, inputOther=200, output=1),
    ])
    entries = _load(sessions_dir, hours_back=1, monkeypatch=monkeypatch)
    assert len(entries) == 1
    assert entries[0].input_tokens == 200


def test_recent_entries_keep_full_session_when_file_is_recent(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "session_s1", [
        _usage_record(3600 * 10, inputOther=100, output=1),
        _usage_record(3600 * 9, inputOther=200, output=1),
    ])
    monkeypatch.setattr(kimi, "SESSIONS_DIR", str(sessions_dir))

    entries = kimi.load_recent_entries(datetime.now(UTC) - timedelta(hours=1))

    assert [entry.input_tokens for entry in entries] == [100, 200]


def test_subagent_wire_and_malformed_rows_ignored(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    wire = _write_session(sessions_dir, "session_s1", [
        _usage_record(60, inputOther=100, output=1),
    ])
    # 子代理的 wire 不算（glob 只认 agents/main/wire.jsonl）
    sub_wire = wire.parents[1] / "agent-0" / "wire.jsonl"
    sub_wire.parent.mkdir(parents=True)
    sub_wire.write_text(json.dumps(_usage_record(50, inputOther=999, output=1)) + "\n", encoding="utf-8")
    # 坏行 / 缺 usage / 坏时间戳不崩、跳过
    with open(wire, "a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps({"type": "usage.record", "model": "kimi-code/k3", "time": _ms(45)}) + "\n")
        f.write(json.dumps({"type": "usage.record", "usage": {"inputOther": 5}, "time": "bad"}) + "\n")

    entries = _load(sessions_dir, monkeypatch=monkeypatch)
    assert len(entries) == 1
    assert entries[0].input_tokens == 100


def test_project_falls_back_to_wd_dir_name(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "wd_wx-clawbot_919bcc6db5be" / "session_s1"
    wire = session_dir / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text(json.dumps(_usage_record(60, inputOther=100, output=1)) + "\n", encoding="utf-8")
    # 无 state.json → 回退 wd_<name>_<hash> 目录名
    entries = _load(sessions_dir, monkeypatch=monkeypatch)
    assert len(entries) == 1
    assert entries[0].project == "wx-clawbot"


def _write_state(sessions_dir: Path, sid: str, updated_at: str, work_dir: str) -> None:
    d = sessions_dir / "wd_proj_abc123def456" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(
        json.dumps({"workDir": work_dir, "updatedAt": updated_at}), encoding="utf-8"
    )


def test_current_session_id_for_cwd_picks_latest_matching(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(kimi, "SESSIONS_DIR", str(sessions_dir))
    monkeypatch.chdir(proj)
    _write_state(sessions_dir, "session_old", "2026-07-24T10:00:00.000Z", str(proj))
    _write_state(sessions_dir, "session_new", "2026-07-24T11:00:00.000Z", str(proj))
    _write_state(sessions_dir, "session_elsewhere", "2026-07-24T12:00:00.000Z", str(tmp_path / "other"))

    assert kimi.current_session_id_for_cwd() == "session_new"


def test_current_session_id_for_cwd_supports_v2_state_schema(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(kimi, "SESSIONS_DIR", str(sessions_dir))
    monkeypatch.chdir(proj)
    session_dir = sessions_dir / "wd_proj_abc123def456" / "session_v2"
    session_dir.mkdir(parents=True)
    updated_at = int(datetime.now(UTC).timestamp() * 1000)
    (session_dir / "state.json").write_text(
        json.dumps({"cwd": str(proj), "updatedAt": updated_at}), encoding="utf-8"
    )

    assert kimi.current_session_id_for_cwd(fresh_within_s=30 * 60) == "session_v2"


def test_current_session_id_for_cwd_freshness_gate(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(kimi, "SESSIONS_DIR", str(sessions_dir))
    monkeypatch.chdir(proj)
    _write_state(sessions_dir, "session_stale", "2020-01-01T00:00:00.000Z", str(proj))

    # 不限新鲜度能命中；限 30 分钟则被门控掉（cli 会话内收窄防常驻误判）
    assert kimi.current_session_id_for_cwd() == "session_stale"
    assert kimi.current_session_id_for_cwd(fresh_within_s=30 * 60) == ""
