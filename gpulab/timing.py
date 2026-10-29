"""Elapsed-time helpers for active runs."""
from datetime import datetime, timezone


def _timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def run_timing(run, at=None):
    """Return elapsed times for the active run and test group."""
    at = at if at is not None else datetime.now(timezone.utc).timestamp()
    started = _timestamp(run.get("started_at"))
    current_started = _timestamp(run.get("current_test_started_at"))
    current_elapsed = max(0, at - current_started) if current_started is not None else None
    return {
        "run_elapsed_seconds": max(0, at - started) if started is not None else None,
        "current_elapsed_seconds": current_elapsed,
    }


def format_duration(seconds):
    if seconds is None:
        return "calculating"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
