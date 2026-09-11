"""Kimi Code CLI 数据源：wire.jsonl 的 usage.record → UsageEntry。

布局：`<kimi_home>/sessions/<wd_*>/<session_*>/agents/main/wire.jsonl`。
`usage.record` 是每 turn 一条的增量记录（`usageScope:"turn"`），字段实测：
`usage.inputOther / output / inputCacheRead / inputCacheCreation`，时间 epoch 毫秒，
`model` 形如 `kimi-code/k3`——语义接近 Claude 的 per-message，一条记录一条 entry。
"""
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .types import AgentInfo, UsageEntry
from .util import (
    file_may_have_events_since,
    iter_jsonl_dicts,
    kimi_home,
    kimi_project_from_session_dir,
    parse_epoch_ms,
)

KIMI_DIR = kimi_home()
SESSIONS_DIR = os.path.join(KIMI_DIR, "sessions")


def detect() -> AgentInfo | None:
    # 以 Kimi 配置根目录判断是否安装；不能要求 sessions/ 已产生，否则裸安装用户无法进入 setup。
    if Path(KIMI_DIR).is_dir():
        return AgentInfo(id="kimi", name="Kimi Code")
    return None


def current_session_id_for_cwd(fresh_within_s: float | None = None) -> str:
    """Kimi 会话内没有 session id 环境变量（实测），回退：取 cwd/workDir 等于当前目录、
    state.json updatedAt 最新的那个 Kimi 会话（兼容 epoch 毫秒与旧版 ISO 字符串）。

    fresh_within_s：只认 updatedAt 在该秒数内的会话（None 不限）。sidebar 启动器在用户
    刚提交提示词后调用、必最新，不限；cli 会话内收窄靠它避免「目录里曾有过 kimi 会话」
    的常驻误判。
    """
    cwd = os.getcwd()
    now = datetime.now(UTC)
    best_id, best_ts = "", None
    for state_path in Path(SESSIONS_DIR).glob("*/session_*/state.json"):
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or (data.get("cwd") or data.get("workDir")) != cwd:
            continue
        ts = _state_updated_at(data.get("updatedAt"))
        if ts is None:
            continue
        if fresh_within_s is not None and (now - ts).total_seconds() > fresh_within_s:
            continue
        if best_ts is None or ts > best_ts:
            best_id, best_ts = state_path.parent.name, ts
    return best_id


def _state_updated_at(raw: object) -> datetime | None:
    if timestamp := parse_epoch_ms(raw):
        return timestamp
    if not isinstance(raw, str):
        return None
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def load_entries(hours_back: int = 0) -> list[UsageEntry]:
    cutoff = None
    if hours_back > 0:
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
    return _load_entries(cutoff, cutoff)


def load_recent_entries(cutoff: datetime) -> list[UsageEntry]:
    """读取 cutoff 后仍有写入的完整会话，供 `tt sessions` 渐进查找最近 N 条。"""
    return _load_entries(cutoff, None)


def _load_entries(file_cutoff: datetime | None, entry_cutoff: datetime | None) -> list[UsageEntry]:
    entries: list[UsageEntry] = []
    seen: set[str] = set()

    sessions_path = Path(SESSIONS_DIR)
    if not sessions_path.is_dir():
        return entries

    for wire_path in sessions_path.glob("*/*/agents/main/wire.jsonl"):
        if not file_may_have_events_since(wire_path, file_cutoff):
            continue
        _parse_wire(wire_path, entries, seen, entry_cutoff)

    entries.sort(key=lambda e: e.timestamp)
    return entries


def _parse_wire(path: Path, entries: list[UsageEntry], seen: set[str], cutoff: datetime | None) -> None:
    # 布局：…/sessions/<wd_*>/<session_*>/agents/main/wire.jsonl
    session_dir = path.parents[2]
    session_id = session_dir.name
    project = kimi_project_from_session_dir(session_dir)
    for data in iter_jsonl_dicts(path):
        if data.get("type") != "usage.record":
            continue
        ts = parse_epoch_ms(data.get("time"))
        if ts is None or (cutoff is not None and ts < cutoff):
            continue
        usage = data.get("usage")
        if not isinstance(usage, dict):
            continue
        entry = _usage_record_to_entry(data, usage, ts, session_id, project)
        if entry is None or entry.dedup_key in seen:
            continue
        seen.add(entry.dedup_key)
        entries.append(entry)


def _usage_record_to_entry(
    data: dict, usage: dict, ts: datetime, session_id: str, project: str
) -> UsageEntry | None:
    input_tokens = _int_or_zero(usage.get("inputOther"))
    output_tokens = _int_or_zero(usage.get("output"))
    cache_read_tokens = _int_or_zero(usage.get("inputCacheRead"))
    cache_creation_tokens = _int_or_zero(usage.get("inputCacheCreation"))
    if not (input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens):
        return None
    model = data.get("model")
    raw_time = data.get("time")
    return UsageEntry(
        timestamp=ts,
        session_id=session_id,
        # usage.record 无消息 id：会话内毫秒时间戳唯一标识一条记录，供 dedup_key 去重
        message_id=f"{session_id}:{raw_time}",
        request_id="",
        model=model if isinstance(model, str) and model else "unknown",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=None,  # wire 不带成本，聚合时走 calculate_cost 定价表
        project=project,
        agent_id="kimi",
    )


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0
