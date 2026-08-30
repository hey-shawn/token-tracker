import os
import sqlite3
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


def _rewrite_virtual_model(model: str) -> str:
    return _VIRTUAL_MODEL_REWRITE.get(model, model)


def detect() -> AgentInfo | None:
    # 以 ~/.codex 目录判断是否安装（与 hooks._has_codex 一致；不要求已产生 sessions/）
    if Path(CODEX_DIR).is_dir():
        return AgentInfo(id="codex", name="Codex")
    return None


def load_entries(hours_back: int = 0) -> list[UsageEntry]:
    entries: list[UsageEntry] = []
    seen: set[str] = set()
    cutoff = None
    if hours_back > 0:
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)

    models = _load_thread_models()

    sessions_path = Path(SESSIONS_DIR)
    if not sessions_path.is_dir():
        return entries

    for jsonl_path in sessions_path.rglob("*.jsonl"):
        if not file_may_have_events_since(jsonl_path, cutoff):
            continue
        _parse_jsonl(jsonl_path, models, entries, seen, cutoff)

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
    session_id = ""
    provider = ""
    last_payload = None
    for data in iter_jsonl_dicts(path):
        if data.get("type") == "session_meta":
            meta = data.get("payload", {})
            session_id = meta.get("id", "")
            p = meta.get("model_provider")
            provider = p if isinstance(p, str) else ""
        if data.get("type") != "event_msg":
            continue
        payload = data.get("payload", {})
        if payload.get("type") != "token_count":
            continue
        rl = payload.get("rate_limits")
        # Spark 等独立池不是账号总 weekly limit，不能覆盖标准 codex 配额。
        if not rl or rl.get("limit_id") != _STANDARD_RATE_LIMIT_ID:
            continue
        event_ts = _parse_event_timestamp(data.get("timestamp"))
        if last_payload is None or event_ts >= last_payload[0]:
            last_payload = (event_ts, rl, payload.get("info") or {}, session_id)

    if not last_payload:
        return None

    event_ts, rl, info, sid = last_payload

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
        provider,
    )


def _parse_event_timestamp(value: object) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _parse_jsonl(
    path: Path,
    models: dict[str, str],
    entries: list[UsageEntry],
    seen: set[str],
    cutoff: datetime | None,
) -> None:
    session_id = ""
    session_ts = ""
    project = "unknown"
    model = "unknown"
    last_usage = None
    msg_count = 0
    session_end_ts: datetime | None = None
    pricing_segments: list[UsageSegment] = []
    seen_segment_totals: set[tuple[int, int, int]] = set()

    for data in iter_jsonl_dicts(path):
        row_type = data.get("type")
        # 取所有事件里最大的 timestamp 作会话结束时间（与 session 开始的差 = 真实跨度，供 sessions 报表）
        ts_raw = data.get("timestamp")
        if ts_raw:
            try:
                et = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if session_end_ts is None or et > session_end_ts:
                    session_end_ts = et
            except (ValueError, AttributeError):
                pass

        if row_type == "session_meta":
            payload = data.get("payload", {})
            session_id = payload.get("id", "")
            session_ts = payload.get("timestamp", "")
            cwd = payload.get("cwd", "")
            if cwd:
                project = project_from_cwd(cwd)
            model = models.get(session_id, "unknown")
            continue

        if row_type != "event_msg":
            continue

        payload = data.get("payload", {})
        if payload.get("type") == "token_count":
            info = payload.get("info")
            if info and info.get("total_token_usage"):
                last_usage = info["total_token_usage"]
                msg_count += 1
                _append_pricing_segment(data, info, pricing_segments, seen_segment_totals)

    if not last_usage or not session_id:
        return

    cached = last_usage.get("cached_input_tokens", 0)
    input_tokens = last_usage.get("input_tokens", 0) - cached
    # reasoning_output_tokens 是 output_tokens 的子集拆分（实测 total_tokens == input + output），不能再加
    output_tokens = last_usage.get("output_tokens", 0)

    if input_tokens == 0 and output_tokens == 0:
        return

    try:
        ts = datetime.fromisoformat(session_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return

    if cutoff and ts < cutoff:
        return

    if session_id in seen:
        return
    seen.add(session_id)

    entries.append(UsageEntry(
        timestamp=ts,
        session_id=session_id,
        message_id=session_id,
        request_id="",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=0,
        cache_read_tokens=cached,
        cost_usd=None,
        project=project,
        agent_id="codex",
        message_count=msg_count,
        session_end=session_end_ts,
        pricing_segments=tuple(pricing_segments),
    ))


def _append_pricing_segment(
    data: dict,
    info: dict,
    segments: list[UsageSegment],
    seen_totals: set[tuple[int, int, int]],
) -> None:
    """Codex token_count 会重复发同一累计快照；按 total 去重后保留每轮 last_token_usage。"""
    total = info.get("total_token_usage")
    last = info.get("last_token_usage")
    if not isinstance(total, dict) or not isinstance(last, dict):
        return
    signature = (
        total.get("input_tokens", 0),
        total.get("cached_input_tokens", 0),
        total.get("output_tokens", 0),
    )
    if signature in seen_totals:
        return
    timestamp = _parse_timestamp(data.get("timestamp"))
    if timestamp is None:
        return
    cached = last.get("cached_input_tokens", 0)
    total_in = last.get("input_tokens", 0)
    output = last.get("output_tokens", 0)
    if not all(isinstance(v, int) and v >= 0 for v in (cached, total_in, output)):
        return
    uncached = max(0, total_in - cached)
    if not (uncached or cached or output):
        return
    seen_totals.add(signature)
    segments.append(UsageSegment(
        timestamp=timestamp,
        input_tokens=uncached,
        output_tokens=output,
        cache_read_tokens=cached,
    ))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
