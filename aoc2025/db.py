"""SQLite database for AoC results storage."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from aoc2025.core import CACHE_DIR, CURRENT_YEAR


def get_db_path(year: int = CURRENT_YEAR) -> Path:
    """Get the database file path."""
    db_dir = CACHE_DIR / str(year)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "aoc.db"


@contextmanager
def get_db(year: int = CURRENT_YEAR):
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path(year))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(year: int = CURRENT_YEAR):
    """Initialize the database schema."""
    with get_db(year) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                day INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                model TEXT,
                label TEXT,
                wall_clock_seconds REAL,
                part1_solved INTEGER DEFAULT 0,
                part2_solved INTEGER DEFAULT 0,
                part1_exec_ms REAL,
                part2_exec_ms REAL,
                timed_out INTEGER DEFAULT 0,
                tokens_in_start INTEGER,
                tokens_out_start INTEGER,
                tokens_cache_read_start INTEGER,
                tokens_cache_create_start INTEGER,
                tokens_in_end INTEGER,
                tokens_out_end INTEGER,
                tokens_cache_read_end INTEGER,
                tokens_cache_create_end INTEGER
            );

            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day INTEGER NOT NULL,
                part INTEGER NOT NULL,
                answer TEXT NOT NULL,
                result TEXT,
                submitted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                day INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (day, key)
            );

            CREATE TABLE IF NOT EXISTS answers (
                day INTEGER NOT NULL,
                input_hash TEXT NOT NULL,
                part INTEGER NOT NULL,
                answer TEXT NOT NULL,
                PRIMARY KEY (day, input_hash, part)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_day ON runs(day);
            CREATE INDEX IF NOT EXISTS idx_submissions_day ON submissions(day, part);
        """)


# --- Runs ---

def create_run(day: int, model: str, label: str, start_tokens: dict, year: int = CURRENT_YEAR) -> dict:
    """Create a new run record."""
    import uuid

    init_db(year)
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.now().isoformat()

    with get_db(year) as conn:
        conn.execute("""
            INSERT INTO runs (id, day, started_at, model, label,
                            tokens_in_start, tokens_out_start,
                            tokens_cache_read_start, tokens_cache_create_start)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, day, started_at, model, label,
            start_tokens.get("input") if start_tokens else None,
            start_tokens.get("output") if start_tokens else None,
            start_tokens.get("cache_read") if start_tokens else None,
            start_tokens.get("cache_create") if start_tokens else None,
        ))

    return {
        "id": run_id,
        "day": day,
        "started_at": started_at,
        "model": model,
        "label": label,
    }


def end_run(run_id: str, end_tokens: dict, year: int = CURRENT_YEAR) -> dict | None:
    """End a run and return the updated record."""
    ended_at = datetime.now().isoformat()

    with get_db(year) as conn:
        conn.execute("""
            UPDATE runs SET
                ended_at = ?,
                tokens_in_end = ?,
                tokens_out_end = ?,
                tokens_cache_read_end = ?,
                tokens_cache_create_end = ?
            WHERE id = ?
        """, (
            ended_at,
            end_tokens.get("input") if end_tokens else None,
            end_tokens.get("output") if end_tokens else None,
            end_tokens.get("cache_read") if end_tokens else None,
            end_tokens.get("cache_create") if end_tokens else None,
            run_id,
        ))

        return get_run(run_id, year)


def update_run(run_id: str, year: int = CURRENT_YEAR, **kwargs):
    """Update run fields."""
    if not kwargs:
        return

    # Map Python bools to SQLite integers
    for key in ["part1_solved", "part2_solved", "timed_out"]:
        if key in kwargs:
            kwargs[key] = 1 if kwargs[key] else 0

    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [run_id]

    with get_db(year) as conn:
        conn.execute(f"UPDATE runs SET {set_clause} WHERE id = ?", values)


def get_run(run_id: str, year: int = CURRENT_YEAR) -> dict | None:
    """Get a run by ID."""
    with get_db(year) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ? OR id LIKE ?",
                          (run_id, f"{run_id}%")).fetchone()
        if row:
            return _row_to_run(row)
    return None


def get_all_runs(year: int = CURRENT_YEAR) -> list[dict]:
    """Get all runs."""
    init_db(year)
    with get_db(year) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        return [_row_to_run(row) for row in rows]


def get_day_runs(day: int, year: int = CURRENT_YEAR) -> list[dict]:
    """Get all runs for a specific day."""
    init_db(year)
    with get_db(year) as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE day = ? ORDER BY started_at", (day,)
        ).fetchall()
        return [_row_to_run(row) for row in rows]


def get_active_run(day: int, year: int = CURRENT_YEAR) -> dict | None:
    """Get the active (not ended) run for a day."""
    with get_db(year) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE day = ? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
            (day,)
        ).fetchone()
        if row:
            return _row_to_run(row)
    return None


def _row_to_run(row: sqlite3.Row) -> dict:
    """Convert a database row to a run dict."""
    run = dict(row)
    # Convert SQLite integers back to bools
    run["part1_solved"] = bool(run.get("part1_solved"))
    run["part2_solved"] = bool(run.get("part2_solved"))
    run["timed_out"] = bool(run.get("timed_out"))
    return run


def get_run_tokens(run: dict) -> dict | None:
    """Calculate token usage for a run (delta between start and end)."""
    if run.get("tokens_in_end") is None or run.get("tokens_in_start") is None:
        return None
    return {
        "input": (run.get("tokens_in_end") or 0) - (run.get("tokens_in_start") or 0),
        "output": (run.get("tokens_out_end") or 0) - (run.get("tokens_out_start") or 0),
        "cache_read": (run.get("tokens_cache_read_end") or 0) - (run.get("tokens_cache_read_start") or 0),
        "cache_create": (run.get("tokens_cache_create_end") or 0) - (run.get("tokens_cache_create_start") or 0),
    }


# --- Submissions ---

def add_submission(day: int, part: int, answer: str, result: str, year: int = CURRENT_YEAR):
    """Record a submission."""
    init_db(year)
    with get_db(year) as conn:
        conn.execute("""
            INSERT INTO submissions (day, part, answer, result, submitted_at)
            VALUES (?, ?, ?, ?, ?)
        """, (day, part, answer, result, datetime.now().isoformat()))


def get_submissions(day: int, year: int = CURRENT_YEAR) -> dict[str, list[dict]]:
    """Get all submissions for a day, grouped by part."""
    init_db(year)
    with get_db(year) as conn:
        rows = conn.execute(
            "SELECT * FROM submissions WHERE day = ? ORDER BY submitted_at",
            (day,)
        ).fetchall()

    result = {"1": [], "2": []}
    for row in rows:
        d = dict(row)
        result[str(d["part"])].append({
            "answer": d["answer"],
            "result": d["result"],
            "submitted_at": d["submitted_at"],
        })
    return result


def is_part_solved(day: int, part: int, year: int = CURRENT_YEAR) -> bool:
    """Check if a part has been solved (has a correct submission)."""
    init_db(year)
    with get_db(year) as conn:
        row = conn.execute("""
            SELECT 1 FROM submissions
            WHERE day = ? AND part = ? AND (result LIKE '%CORRECT%' OR result LIKE '%Already completed%')
            LIMIT 1
        """, (day, part)).fetchone()
        return row is not None


# --- Metrics ---

def set_metric(day: int, key: str, value: Any, year: int = CURRENT_YEAR):
    """Set a metric value (JSON-serialized)."""
    init_db(year)
    with get_db(year) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO metrics (day, key, value, updated_at)
            VALUES (?, ?, ?, ?)
        """, (day, key, json.dumps(value), datetime.now().isoformat()))


def get_metric(day: int, key: str, year: int = CURRENT_YEAR) -> Any:
    """Get a metric value."""
    init_db(year)
    with get_db(year) as conn:
        row = conn.execute(
            "SELECT value FROM metrics WHERE day = ? AND key = ?",
            (day, key)
        ).fetchone()
        if row:
            return json.loads(row["value"])
    return None


def get_all_metrics(day: int, year: int = CURRENT_YEAR) -> dict:
    """Get all metrics for a day."""
    init_db(year)
    with get_db(year) as conn:
        rows = conn.execute(
            "SELECT key, value FROM metrics WHERE day = ?", (day,)
        ).fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}


# --- Answers ---

def save_answer(day: int, input_hash: str, part: int, answer: str, year: int = CURRENT_YEAR):
    """Save a known correct answer for an input."""
    init_db(year)
    with get_db(year) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO answers (day, input_hash, part, answer)
            VALUES (?, ?, ?, ?)
        """, (day, input_hash, part, answer))


def get_answers(day: int, year: int = CURRENT_YEAR) -> dict[str, dict[str, str]]:
    """Get all known answers for a day, keyed by input hash."""
    init_db(year)
    with get_db(year) as conn:
        rows = conn.execute(
            "SELECT input_hash, part, answer FROM answers WHERE day = ?",
            (day,)
        ).fetchall()

    result = {}
    for row in rows:
        h = row["input_hash"]
        if h not in result:
            result[h] = {}
        result[h][str(row["part"])] = row["answer"]
    return result


# --- Run artifacts (still file-based for large content) ---

def get_run_dir(run_id: str, year: int = CURRENT_YEAR) -> Path:
    """Get the directory for a run's artifacts."""
    return CACHE_DIR / str(year) / "runs" / run_id


def save_run_artifact(run_id: str, filename: str, content: str, year: int = CURRENT_YEAR):
    """Save an artifact for a run."""
    run_dir = get_run_dir(run_id, year)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(content)


def get_run_artifact(run_id: str, filename: str, year: int = CURRENT_YEAR) -> str | None:
    """Get an artifact for a run."""
    path = get_run_dir(run_id, year) / filename
    if path.exists():
        return path.read_text()
    return None


# --- Migration from JSON files ---

def migrate_from_json(year: int = CURRENT_YEAR):
    """Migrate data from old JSON files to SQLite."""
    init_db(year)

    # Migrate runs.json
    runs_file = CACHE_DIR / str(year) / "runs.json"
    if runs_file.exists():
        runs = json.loads(runs_file.read_text())
        with get_db(year) as conn:
            for run in runs:
                start_tokens = run.get("start_tokens", {})
                end_tokens = run.get("end_tokens", {})
                conn.execute("""
                    INSERT OR IGNORE INTO runs
                    (id, day, started_at, ended_at, model, label, wall_clock_seconds,
                     part1_solved, part2_solved, part1_exec_ms, part2_exec_ms, timed_out,
                     tokens_in_start, tokens_out_start, tokens_cache_read_start, tokens_cache_create_start,
                     tokens_in_end, tokens_out_end, tokens_cache_read_end, tokens_cache_create_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run["id"], run["day"], run["started_at"], run.get("ended_at"),
                    run.get("model"), run.get("label"), run.get("wall_clock_seconds"),
                    1 if run.get("part1_solved") else 0,
                    1 if run.get("part2_solved") else 0,
                    run.get("part1_exec_ms") or run.get("part1_time_ms"),
                    run.get("part2_exec_ms") or run.get("part2_time_ms"),
                    1 if run.get("timed_out") else 0,
                    start_tokens.get("input"), start_tokens.get("output"),
                    start_tokens.get("cache_read"), start_tokens.get("cache_create"),
                    end_tokens.get("input"), end_tokens.get("output"),
                    end_tokens.get("cache_read"), end_tokens.get("cache_create"),
                ))
        print(f"Migrated {len(runs)} runs from runs.json")

    # Migrate per-day JSON files
    for day_dir in (CACHE_DIR / str(year)).iterdir():
        if not day_dir.is_dir() or not day_dir.name.isdigit():
            continue
        day = int(day_dir.name)

        # Migrate submissions.json
        subs_file = day_dir / "submissions.json"
        if subs_file.exists():
            subs = json.loads(subs_file.read_text())
            with get_db(year) as conn:
                for part, part_subs in subs.items():
                    for sub in part_subs:
                        conn.execute("""
                            INSERT INTO submissions (day, part, answer, result, submitted_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (day, int(part), sub["answer"], sub.get("result", ""),
                              sub.get("submitted_at", datetime.now().isoformat())))
            print(f"Migrated submissions for day {day}")

        # Migrate metrics.json
        metrics_file = day_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            for key, value in metrics.items():
                set_metric(day, key, value, year)
            print(f"Migrated metrics for day {day}")

        # Migrate answers.json
        answers_file = day_dir / "answers.json"
        if answers_file.exists():
            answers = json.loads(answers_file.read_text())
            for input_hash, parts in answers.items():
                for part, answer in parts.items():
                    save_answer(day, input_hash, int(part), answer, year)
            print(f"Migrated answers for day {day}")
