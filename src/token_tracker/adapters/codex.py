import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .types import AgentInfo, RateLimits, UsageEntry, UsageSegment, normalize_pct
from .util import codex_home, file_may_have_events_since, iter_jsonl_dicts, project_from_cwd

CODEX_DIR = codex_home()
SESSIONS_DIR = os.path.join(CODEX_DIR, "sessions")
STATE_DB = os.path.join(CODEX_DIR, "state_5.sqlite")
_RATE_LIMIT_SCAN_FILES = 5  # 只扫最近改动的 N 个 session 文件找限额信息
_STANDARD_RATE_LIMIT_ID = "codex"

# Codex 内部虚拟 model 改写到背后真实 model，避免它们在 Model Trend / sessions 等报表里独占行；
# 同时让 cost 走真实 model 的精确定价（cost.py 的 codex- 系列兜底是双保险）。
# 已知虚拟 model：
#   codex-auto-review —— stop-time auto-review gate，背后跑当前主 codex 模型（gpt-5.5）
_VIRTUAL_MODEL_REWRITE = {
    "codex-auto-review": "gpt-5.5",
}


@dataclass
class CodexSessionSnapshot:
    """单次扫描得到的当前会话快照，供高频 statusline 复用。"""

    session_id: str = ""
    cwd: str = ""
    info: dict | None = None
    model: str = ""
    effort: str = ""
    provider: str = ""
    rate_limits: RateLimits | None = None
    usage_entry: UsageEntry | None = None


@dataclass
class _SessionData:
    session_id: str = ""
    session_ts: str = ""
    cwd: str = ""
    provider: str = ""
    model: str = ""
    effort: str = ""
    last_info: dict | None = None
    msg_count: int = 0
    session_end: datetime | None = None
    pricing_segments: list[UsageSegment] = field(default_factory=list)
    seen_segment_totals: set[tuple[int, int, int, int]] = field(default_factory=set)
    last_rate_payload: tuple[float, dict, dict, str] | None = None


def _rewrite_virtual_model(model: str) -> str:
    return _VIRTUAL_MODEL_REWRITE.get(model, model)


def detect() -> AgentInfo | None:
    # 以 ~/.codex 目录判断是否安装（与 hooks._has_codex 一致；不要求已产生 sessions/）
    if Path(CODEX_DIR).is_dir():
        return AgentInfo(id="codex", name="Codex")
    return None


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

    models = _load_thread_models()

    sessions_path = Path(SESSIONS_DIR)
    if not sessions_path.is_dir():
        return entries

    for jsonl_path in sessions_path.rglob("*.jsonl"):
        if not file_may_have_events_since(jsonl_path, file_cutoff):
            continue
        _parse_jsonl(jsonl_path, models, entries, seen, entry_cutoff)

    entries.sort(key=lambda e: e.timestamp)
    return entries


def _load_thread_models() -> dict[str, str]:
    if not os.path.exists(STATE_DB):
        return {}
    try:
        conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
        rows = conn.execute("SELECT id, model FROM threads WHERE model IS NOT NULL").fetchall()
        conn.close()
        return {row[0]: _rewrite_virtual_model(row[1]) for row in rows}
    except (sqlite3.Error, OSError):
        return {}


def load_rate_limits(provider: str | None = None) -> RateLimits | None:
    """最近会话的账号限额快照。

    provider 给出时只采纳 session_meta.model_provider 相同的会话——同一 CODEX_HOME 下
    多账号 / 多 model_provider（如 codex --profile deepseek）混跑时，不能把别的
    provider 的配额（或 openai 配额）显示在当前会话上。
    """
    sessions_path = Path(SESSIONS_DIR)
    if not sessions_path.is_dir():
        return None

    # session 文件在轮转，rglob 与 stat 之间文件可能消失：mtime 取不到时退化为 0，避免整体崩溃
    jsonl_files = sorted(sessions_path.rglob("*.jsonl"), key=_safe_mtime, reverse=True)
    models = _load_thread_models()

    latest_snapshot: tuple[float, RateLimits] | None = None
    for path in jsonl_files[:_RATE_LIMIT_SCAN_FILES]:
        snapshot = _extract_rate_limits_snapshot(path, models)
        if not snapshot:
            continue
        if provider is not None and snapshot[2] != provider:
            continue
        if latest_snapshot is None or snapshot[0] > latest_snapshot[0]:
            latest_snapshot = snapshot[:2]
    return latest_snapshot[1] if latest_snapshot else None


def load_session_rate_limits(path: Path | str) -> RateLimits | None:
    """单个会话文件自己的限额快照（statusline 优先用当前会话，避免串账号）。"""
    return _extract_rate_limits(Path(path), {})


def session_provider(path: Path | str) -> str:
    """会话的 model_provider（session_meta）；读不到返回 ""。"""
    for data in iter_jsonl_dicts(Path(path)):
        if data.get("type") == "session_meta":
            provider = data.get("payload", {}).get("model_provider")
            return provider if isinstance(provider, str) else ""
    return ""


def latest_session_provider() -> str:
    """最近活跃会话的 model_provider（CLI 面板 Limit 卡片按它过滤，混跑时不显示错账号配额）。

    只扫最近改动的几个文件，找到第一个带 session_meta.model_provider 的即返回。
    """
    sessions_path = Path(SESSIONS_DIR)
    if not sessions_path.is_dir():
        return ""
    jsonl_files = sorted(sessions_path.rglob("*.jsonl"), key=_safe_mtime, reverse=True)
    for path in jsonl_files[:_RATE_LIMIT_SCAN_FILES]:
        provider = session_provider(path)
        if provider:
            return provider
    return ""


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _extract_rate_limits(path: Path, models: dict[str, str]) -> RateLimits | None:
    snapshot = _extract_rate_limits_snapshot(path, models)
    return snapshot[1] if snapshot else None


def _extract_rate_limits_snapshot(path: Path, models: dict[str, str]) -> tuple[float, RateLimits, str] | None:
    data = _read_session_data(path)
    return _rate_limits_from_data(data, models)


def _rate_limits_from_data(data: _SessionData, models: dict[str, str]) -> tuple[float, RateLimits, str] | None:
    if not data.last_rate_payload:
        return None

    event_ts, rl, info, sid = data.last_rate_payload

    now_ts = datetime.now(UTC).timestamp()
    five_pct = five_reset = None
    seven_pct = seven_reset = None

    # 按 window_minutes 字段分配 5h / 7d 桶，
    # 而不是固定 primary→5h、secondary→7d（free plan 实测 primary 为 7 天窗口）
    for bucket in (rl.get("primary"), rl.get("secondary")):
        if not bucket:
            continue
        resets = bucket.get("resets_at")
        window = bucket.get("window_minutes") or 0
        pct = normalize_pct(bucket.get("used_percent"), resets, now_ts)
        if window < 1440:
            five_pct, five_reset = pct, resets
        else:
            seven_pct, seven_reset = pct, resets

    if five_pct is None and seven_pct is None:
        return None

    return (
        event_ts,
        RateLimits(
            five_hour_pct=five_pct,
            five_hour_resets_at=five_reset,
            seven_day_pct=seven_pct,
            seven_day_resets_at=seven_reset,
            model=models.get(sid, ""),
            plan_type=rl.get("plan_type") or "",
            context_window=info.get("model_context_window"),
        ),
        data.provider,
    )


def _parse_jsonl(
    path: Path,
    models: dict[str, str],
    entries: list[UsageEntry],
    seen: set[str],
    cutoff: datetime | None,
) -> None:
    data = _read_session_data(path)
    entry = _usage_entry_from_data(data, models)
    if entry is None or (cutoff and entry.timestamp < cutoff) or entry.session_id in seen:
        return
    seen.add(entry.session_id)
    entries.append(entry)


def load_session_snapshot(path: Path | str) -> CodexSessionSnapshot:
    """只扫描一次会话文件，同时生成 statusline 所需元数据、限额与计价 entry。"""
    data = _read_session_data(Path(path))
    model = _rewrite_virtual_model(data.model)
    models = {data.session_id: model} if model else {}
    rate_snapshot = _rate_limits_from_data(data, models)
    return CodexSessionSnapshot(
        session_id=data.session_id,
        cwd=data.cwd,
        info=data.last_info,
        model=model,
        effort=data.effort,
        provider=data.provider,
        rate_limits=rate_snapshot[1] if rate_snapshot else None,
        usage_entry=_usage_entry_from_data(data, models),
    )


def _read_session_data(path: Path) -> _SessionData:
    state = _SessionData()
    for data in iter_jsonl_dicts(path):
        row_type = data.get("type")
        # 取所有事件里最大的 timestamp 作会话结束时间（与 session 开始的差 = 真实跨度，供 sessions 报表）
        event_time = _parse_timestamp(data.get("timestamp"))
        if event_time is not None and (state.session_end is None or event_time > state.session_end):
            state.session_end = event_time

        payload = data.get("payload")
        if not isinstance(payload, dict):
            continue

        if row_type == "session_meta":
            state.session_id = payload.get("id", "") or state.session_id
            state.session_ts = payload.get("timestamp", "") or state.session_ts
            state.cwd = payload.get("cwd", "") or state.cwd
            provider = payload.get("model_provider")
            if isinstance(provider, str) and provider:
                state.provider = provider
            continue

        if row_type == "turn_context":
            model = payload.get("model")
            effort = payload.get("effort")
            if isinstance(model, str) and model:
                state.model = model
            if isinstance(effort, str) and effort:
                state.effort = effort
            continue

        if row_type != "event_msg":
            continue

        if payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                state.last_info = info
                state.msg_count += 1
                _append_pricing_segment(data, info, state.pricing_segments, state.seen_segment_totals)
            rl = payload.get("rate_limits")
            # Spark 等独立池不是账号总 weekly limit，不能覆盖标准 codex 配额。
            if isinstance(rl, dict) and rl.get("limit_id") == _STANDARD_RATE_LIMIT_ID:
                event_ts = event_time.timestamp() if event_time is not None else 0.0
                if state.last_rate_payload is None or event_ts >= state.last_rate_payload[0]:
                    state.last_rate_payload = (
                        event_ts,
                        rl,
                        info if isinstance(info, dict) else {},
                        state.session_id,
                    )
    return state


def _usage_tokens(usage: dict) -> tuple[int, int, int, int] | None:
    """Codex input 包含缓存读写；拆为互斥桶，避免重复计数与缓存写入少计价。"""
    total_in = usage.get("input_tokens", 0)
    output = usage.get("output_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    written = usage.get("cache_write_input_tokens", 0)
    if not all(type(v) is int and v >= 0 for v in (total_in, output, cached, written)):
        return None
    if cached + written > total_in:
        return None
    return total_in - cached - written, output, written, cached


def _usage_entry_from_data(data: _SessionData, models: dict[str, str]) -> UsageEntry | None:
    last_usage = (data.last_info or {}).get("total_token_usage")
    if not isinstance(last_usage, dict) or not data.session_id:
        return None

    counts = _usage_tokens(last_usage)
    if counts is None or not any(counts):
        return None
    input_tokens, output_tokens, written, cached = counts
    # reasoning_output_tokens 是 output_tokens 的子集拆分（实测 total_tokens == input + output），不能再加

    try:
        ts = datetime.fromisoformat(data.session_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    model = models.get(data.session_id) or _rewrite_virtual_model(data.model) or "unknown"
    project = project_from_cwd(data.cwd) if data.cwd else "unknown"
    return UsageEntry(
        timestamp=ts,
        session_id=data.session_id,
        message_id=data.session_id,
        request_id="",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=written,
        cache_read_tokens=cached,
        cost_usd=None,
        project=project,
        agent_id="codex",
        message_count=data.msg_count,
        session_end=data.session_end,
        pricing_segments=tuple(data.pricing_segments),
    )


def _append_pricing_segment(
    data: dict,
    info: dict,
    segments: list[UsageSegment],
    seen_totals: set[tuple[int, int, int, int]],
) -> None:
    """Codex token_count 会重复发同一累计快照；按 total 去重后保留每轮 last_token_usage。"""
    total = info.get("total_token_usage")
    last = info.get("last_token_usage")
    if not isinstance(total, dict) or not isinstance(last, dict):
        return
    signature = _usage_tokens(total)
    if signature is None or signature in seen_totals:
        return
    timestamp = _parse_timestamp(data.get("timestamp"))
    if timestamp is None:
        return
    counts = _usage_tokens(last)
    if counts is None or not any(counts):
        return
    uncached, output, written, cached = counts
    seen_totals.add(signature)
    segments.append(UsageSegment(
        timestamp=timestamp,
        input_tokens=uncached,
        output_tokens=output,
        cache_creation_tokens=written,
        cache_read_tokens=cached,
    ))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
