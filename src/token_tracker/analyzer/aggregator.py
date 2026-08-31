from collections import defaultdict
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from ..adapters.types import DailyStats, MonthlyStats, SessionStats, UsageEntry, WeeklyStats
from ..tz import system_tz
from .cost import calculate_cost


def add_token_fields(target: Any, e: UsageEntry, cost: float) -> None:
    """把一条 UsageEntry 的 6 个通用 token/成本字段累加到任意 *Stats / StatusSummary 上。"""
    target.input_tokens += e.input_tokens
    target.output_tokens += e.output_tokens
    target.cache_creation_tokens += e.cache_creation_tokens
    target.cache_read_tokens += e.cache_read_tokens
    target.total_tokens += e.total_tokens
    target.cost_usd += cost


def _aggregate_by_key(
    entries: list[UsageEntry],
    key_fn: Callable[[UsageEntry], str],
    factory: Callable[[str, UsageEntry], Any],
    sort_key: str,
) -> list[Any]:
    """按 key_fn 分桶累加（daily/weekly/monthly 共用骨架）。

    factory(key, entry) 负责构造对应的 *Stats（weekly 需借 entry 算周起止）。
    """
    by_key: dict[str, Any] = {}
    sessions_by_key: dict[str, set[str]] = defaultdict(set)

    for e in entries:
        k = key_fn(e)
        s = by_key.get(k)
        if s is None:
            s = by_key[k] = factory(k, e)
        add_token_fields(s, e, calculate_cost(e))
        s.message_count += e.message_count
        s.models[e.model] = s.models.get(e.model, 0) + e.total_tokens
        s.projects[e.project] = s.projects.get(e.project, 0) + e.total_tokens
        sessions_by_key[k].add(e.session_id)

    for k, sessions in sessions_by_key.items():
        by_key[k].session_count = len(sessions)

    return sorted(by_key.values(), key=lambda s: getattr(s, sort_key))


# daily/weekly/monthly 一律按**系统时区**的日历日分桶（entries 时间戳是 UTC；
# 不转换会把北京时间 00:00-08:00 的用量记到前一天，且与 tt status 的「今天」口径打架）。
# tz 在函数入口取一次（system_tz 每次调用要 readlink），闭包复用。

def aggregate_daily(entries: list[UsageEntry]) -> list[DailyStats]:
    tz = system_tz()
    return _aggregate_by_key(
        entries,
        lambda e: e.timestamp.astimezone(tz).date().isoformat(),
        lambda k, e: DailyStats(date=k),
        "date",
    )


def aggregate_monthly(entries: list[UsageEntry]) -> list[MonthlyStats]:
    tz = system_tz()

    def _month_key(e: UsageEntry) -> str:
        local = e.timestamp.astimezone(tz)
        return f"{local.year:04d}-{local.month:02d}"

    return _aggregate_by_key(
        entries,
        _month_key,
        lambda k, e: MonthlyStats(month=k),
        "month",
    )


def aggregate_weekly(entries: list[UsageEntry]) -> list[WeeklyStats]:
    tz = system_tz()

    def _week_key(e: UsageEntry) -> str:
        d = e.timestamp.astimezone(tz).date()
        return (d - timedelta(days=d.weekday())).isoformat()

    def _factory(k: str, e: UsageEntry) -> WeeklyStats:
        monday = date.fromisoformat(k)
        sunday = monday + timedelta(days=6)
        return WeeklyStats(week=k, week_start=monday.strftime("%m-%d"), week_end=sunday.strftime("%m-%d"))

    return _aggregate_by_key(entries, _week_key, _factory, "week")


ACTIVE_GAP_CAP_MIN = 30  # 相邻 entry 间隔超过此值视为「人离开」，不计入活跃时长


def aggregate_sessions(entries: list[UsageEntry]) -> list[SessionStats]:
    by_session: dict[str, list[UsageEntry]] = defaultdict(list)

    for e in entries:
        by_session[e.session_id].append(e)

    sessions: list[SessionStats] = []
    for session_id, session_entries in by_session.items():
        session_entries.sort(key=lambda e: e.timestamp)
        first = session_entries[0]
        last = session_entries[-1]
        # codex 单条 entry 自带会话结束时间（session_end）；取它与末条时间的最大值作真实结束，算出跨度
        ends = [e.session_end for e in session_entries if e.session_end]
        end_ts = max([last.timestamp, *ends])
        duration = (end_ts - first.timestamp).total_seconds() / 60
        # 活跃时长：累加相邻间隔，但单段间隔超过 CAP 视为离开、整段丢弃
        active = 0.0
        for a, b in zip(session_entries, session_entries[1:], strict=False):
            gap = (b.timestamp - a.timestamp).total_seconds() / 60
            if gap <= ACTIVE_GAP_CAP_MIN:
                active += gap
        # codex 会话是单条 entry（段内无相邻间隔可累计），活跃时长退化为整段跨度——
        # 否则恒为 0，被 status 的 active_minutes>=5 过滤，codex 会话永远进不了当天列表
        if len(session_entries) == 1 and first.session_end:
            active = duration

        # 代表模型按 output_tokens（真实生成量）选，output 持平时用 total 兜底；
        # 不用 total 直接选，避免后台小模型（如 Haiku）读了大量上下文、cache_read 撑高 total 被误判为主力。
        out_by_model: dict[str, int] = defaultdict(int)
        tot_by_model: dict[str, int] = defaultdict(int)
        for e in session_entries:
            out_by_model[e.model] += e.output_tokens
            tot_by_model[e.model] += e.total_tokens
        ranked = sorted(out_by_model, key=lambda m: (out_by_model[m], tot_by_model[m]), reverse=True)
        primary_model = ranked[0] if ranked else "unknown"

        s = SessionStats(
            session_id=session_id,
            project=first.project,
            model=primary_model,
            start_time=first.timestamp,
            end_time=end_ts,
            duration_minutes=round(duration, 1),
            active_minutes=round(active, 1),
            models={m: out_by_model[m] for m in ranked},
        )
        for e in session_entries:
            add_token_fields(s, e, calculate_cost(e))
            s.message_count += e.message_count

        sessions.append(s)

    sessions.sort(key=lambda s: s.start_time, reverse=True)
    return sessions
