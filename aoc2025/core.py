"""Core functionality for fetching problems, inputs, and submitting answers."""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify

# Load .env file if present
load_dotenv()

BASE_URL = "https://adventofcode.com"
CURRENT_YEAR = 2025


def _get_cache_root() -> Path:
    """Get the cache root directory from env or default."""
    if cache_dir := os.environ.get("AOC_CACHE_DIR"):
        return Path(cache_dir)
    return Path.home() / ".cache" / "aoc2025"


CACHE_DIR = _get_cache_root()


def input_hash(content: str) -> str:
    """Get a short hash of input content for content-addressable storage."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_session() -> str:
    """Get the AoC session cookie from environment."""
    session = os.environ.get("AOC_SESSION")
    if not session:
        print("Error: AOC_SESSION environment variable not set.", file=sys.stderr)
        print("Set it in .env or export AOC_SESSION=your_cookie", file=sys.stderr)
        sys.exit(1)
    return session


def get_cache_dir(day: int, year: int = CURRENT_YEAR) -> Path:
    """Get the cache directory for a specific day."""
    cache_dir = CACHE_DIR / str(year) / str(day)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fetch_problem(day: int, year: int = CURRENT_YEAR, force: bool = False) -> str:
    """Fetch the problem statement for a day, using cache if available."""
    cache_dir = get_cache_dir(day, year)
    cache_file = cache_dir / "problem.md"

    if cache_file.exists() and not force:
        return cache_file.read_text()

    url = f"{BASE_URL}/{year}/day/{day}"
    response = requests.get(url, cookies={"session": get_session()})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("main")

    if not article:
        print(f"Error: Could not find problem content for day {day}", file=sys.stderr)
        sys.exit(1)

    markdown = markdownify(str(article), heading_style="ATX")

    while "\n\n\n" in markdown:
        markdown = markdown.replace("\n\n\n", "\n\n")

    cache_file.write_text(markdown)
    print(f"Cached problem to {cache_file}")
    return markdown


def get_inputs_dir(day: int, year: int = CURRENT_YEAR) -> Path:
    """Get the inputs directory for a specific day."""
    inputs_dir = get_cache_dir(day, year) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    return inputs_dir


def store_input(day: int, content: str, year: int = CURRENT_YEAR) -> str:
    """Store an input by its content hash. Returns the hash."""
    h = input_hash(content)
    inputs_dir = get_inputs_dir(day, year)
    input_file = inputs_dir / h
    if not input_file.exists():
        input_file.write_text(content)
    return h


def get_input_by_hash(day: int, h: str, year: int = CURRENT_YEAR) -> str:
    """Retrieve an input by its hash."""
    inputs_dir = get_inputs_dir(day, year)
    input_file = inputs_dir / h
    if not input_file.exists():
        raise FileNotFoundError(f"No input with hash {h} for day {day}")
    return input_file.read_text()


def list_inputs(day: int, year: int = CURRENT_YEAR) -> list[str]:
    """List all stored input hashes for a day."""
    inputs_dir = get_inputs_dir(day, year)
    if not inputs_dir.exists():
        return []
    return [f.name for f in inputs_dir.iterdir() if f.is_file()]


def fetch_input(day: int, year: int = CURRENT_YEAR, force: bool = False) -> str:
    """Fetch the puzzle input for a day, using cache if available."""
    cache_dir = get_cache_dir(day, year)
    ref_file = cache_dir / "input_hash.txt"

    if ref_file.exists() and not force:
        h = ref_file.read_text().strip()
        return get_input_by_hash(day, h, year)

    url = f"{BASE_URL}/{year}/day/{day}/input"
    response = requests.get(url, cookies={"session": get_session()})
    response.raise_for_status()

    input_text = response.text
    h = store_input(day, input_text, year)
    ref_file.write_text(h)
    print(f"Cached input with hash {h}")
    return input_text


def fetch(day: int, year: int = CURRENT_YEAR, force: bool = False) -> tuple[str, str]:
    """Fetch both problem statement and input for a day."""
    record_started(day, year)
    problem = fetch_problem(day, year, force)
    input_text = fetch_input(day, year, force)
    return problem, input_text


def get_submissions(day: int, year: int = CURRENT_YEAR) -> dict:
    """Get the submission history for a day."""
    cache_dir = get_cache_dir(day, year)
    submissions_file = cache_dir / "submissions.json"

    if submissions_file.exists():
        return json.loads(submissions_file.read_text())
    return {"1": [], "2": []}


def save_submissions(day: int, submissions: dict, year: int = CURRENT_YEAR):
    """Save the submission history for a day."""
    cache_dir = get_cache_dir(day, year)
    submissions_file = cache_dir / "submissions.json"
    submissions_file.write_text(json.dumps(submissions, indent=2))


def get_answers(day: int, year: int = CURRENT_YEAR) -> dict[str, dict[str, str]]:
    """Get all cached correct answers for a day, keyed by input hash.

    Returns: {"abc123...": {"1": "ans1", "2": "ans2"}, ...}
    """
    cache_dir = get_cache_dir(day, year)
    answers_file = cache_dir / "answers.json"

    if answers_file.exists():
        return json.loads(answers_file.read_text())
    return {}


def save_answer(
    day: int,
    part: int,
    answer: str,
    input_content: str,
    year: int = CURRENT_YEAR,
):
    """Save a correct answer to the cache for a specific input."""
    cache_dir = get_cache_dir(day, year)
    answers_file = cache_dir / "answers.json"

    h = store_input(day, input_content, year)

    all_answers = get_answers(day, year)
    if h not in all_answers:
        all_answers[h] = {}
    all_answers[h][str(part)] = answer
    answers_file.write_text(json.dumps(all_answers, indent=2))


def get_primary_input_hash(day: int, year: int = CURRENT_YEAR) -> str | None:
    """Get the hash of the primary (fetched) input for a day."""
    cache_dir = get_cache_dir(day, year)
    ref_file = cache_dir / "input_hash.txt"
    if ref_file.exists():
        return ref_file.read_text().strip()
    return None


def read_input(day: int, year: int = CURRENT_YEAR) -> str:
    """Read the primary cached input for a day."""
    h = get_primary_input_hash(day, year)
    if h is None:
        return fetch_input(day, year)
    return get_input_by_hash(day, h, year)


def show(day: int, year: int = CURRENT_YEAR) -> str:
    """Display the cached problem statement."""
    cache_dir = get_cache_dir(day, year)
    cache_file = cache_dir / "problem.md"

    if not cache_file.exists():
        print(f"No cached problem for day {day}. Run 'fetch {day}' first.")
        return ""

    return cache_file.read_text()


def extract_examples(day: int, year: int = CURRENT_YEAR) -> list[dict]:
    """Extract example inputs and expected outputs from problem.md.

    Returns list of {"input": str, "part1": str|None, "part2": str|None}
    """
    import re

    cache_dir = get_cache_dir(day, year)
    problem_file = cache_dir / "problem.md"

    if not problem_file.exists():
        return []

    content = problem_file.read_text()

    # Split into Part 1 and Part 2 sections
    parts = re.split(r'## --- Part Two ---', content)
    part1_text = parts[0] if parts else ""
    part2_text = parts[1] if len(parts) > 1 else ""

    examples = []

    # Find code blocks (``` ... ```)
    code_blocks = re.findall(r'```\n(.*?)\n```', content, re.DOTALL)

    # Find expected answers - look for patterns like:
    # "is `X`", "would be `X`", "produces `X`", "total of `X`", "answer is `X`"
    # Usually the last number mentioned before "Part Two" or end of Part 1
    def find_answer(text):
        # Look for explicit answer patterns
        patterns = [
            r'(?:is|would be|produces|equals|total[^`]*?|answer[^`]*?)[^`]*`(\d+)`',
            r'`(\d+)`[^`]*?(?:paths|times|total|answer)',
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        return matches[-1] if matches else None

    part1_answer = find_answer(part1_text)
    part2_answer = find_answer(part2_text) if part2_text else None

    # The first substantial code block is usually the example input
    for block in code_blocks:
        block = block.strip()
        if len(block) > 10 and '\n' in block:  # Substantial multi-line block
            examples.append({
                "input": block,
                "part1": part1_answer,
                "part2": part2_answer,
            })
            break  # Usually just one main example

    return examples


def get_claude_token_counts() -> dict | None:
    """Read current cumulative token counts from Claude Code conversation logs.

    Parses the .jsonl conversation files in ~/.claude/projects/ to get accurate
    per-message token usage data.
    """
    # Find the project directory for current working directory
    cwd = Path.cwd()
    claude_projects = Path.home() / ".claude" / "projects"

    if not claude_projects.exists():
        return None

    # Convert cwd to Claude's directory naming convention
    # e.g., /home/user/projects/foo -> -home-user-projects-foo
    cwd_slug = str(cwd).replace("/", "-")
    if cwd_slug.startswith("-"):
        cwd_slug = cwd_slug  # Keep leading dash

    project_dir = claude_projects / cwd_slug

    if not project_dir.exists():
        # Try to find a matching directory
        for d in claude_projects.iterdir():
            if d.is_dir() and str(cwd).replace("/", "-").lstrip("-") in d.name:
                project_dir = d
                break
        else:
            return None

    # Find and parse all .jsonl conversation files
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}

    for jsonl_file in project_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "message" in obj and isinstance(obj["message"], dict):
                            usage = obj["message"].get("usage", {})
                            if usage:
                                totals["input"] += usage.get("input_tokens", 0)
                                totals["output"] += usage.get("output_tokens", 0)
                                totals["cache_read"] += usage.get("cache_read_input_tokens", 0)
                                totals["cache_create"] += usage.get("cache_creation_input_tokens", 0)
                    except json.JSONDecodeError:
                        continue
        except (IOError, OSError):
            continue

    # Return None if no data found
    if totals["input"] == 0 and totals["output"] == 0:
        return None

    return totals


def get_metrics(day: int, year: int = CURRENT_YEAR) -> dict:
    """Get metrics for a day."""
    cache_dir = get_cache_dir(day, year)
    metrics_file = cache_dir / "metrics.json"

    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return {}


def save_metrics(day: int, metrics: dict, year: int = CURRENT_YEAR):
    """Save metrics for a day."""
    cache_dir = get_cache_dir(day, year)
    metrics_file = cache_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))


def get_current_model() -> str | None:
    """Try to detect the current Claude model from environment or conversation logs."""
    import os

    # Check environment variable first
    if model := os.environ.get("CLAUDE_MODEL"):
        return model

    # Try to extract from most recent conversation log
    claude_projects = Path.home() / ".claude" / "projects"
    cwd = Path.cwd()
    cwd_slug = str(cwd).replace("/", "-")

    project_dir = claude_projects / cwd_slug
    if not project_dir.exists():
        for d in claude_projects.iterdir():
            if d.is_dir() and str(cwd).replace("/", "-").lstrip("-") in d.name:
                project_dir = d
                break
        else:
            return None

    # Find most recent jsonl file
    jsonl_files = list(project_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    latest_file = max(jsonl_files, key=lambda f: f.stat().st_mtime)

    # Read last few lines looking for model info
    try:
        with open(latest_file) as f:
            lines = f.readlines()
            for line in reversed(lines[-50:]):  # Check last 50 messages
                obj = json.loads(line)
                if "message" in obj and isinstance(obj["message"], dict):
                    if model := obj["message"].get("model"):
                        return model
    except (json.JSONDecodeError, IOError):
        pass

    return None


def record_started(day: int, year: int = CURRENT_YEAR):
    """Record when work on a day started (first fetch)."""
    metrics = get_metrics(day, year)
    if "started_at" not in metrics:
        metrics["started_at"] = datetime.now().isoformat()
        tokens = get_claude_token_counts()
        if tokens:
            metrics["started_tokens"] = tokens
        if model := get_current_model():
            metrics["model"] = model
        save_metrics(day, metrics, year)


def record_solved(day: int, part: int, year: int = CURRENT_YEAR):
    """Record when a part was solved."""
    metrics = get_metrics(day, year)
    key = f"part{part}_solved_at"
    if key not in metrics:
        metrics[key] = datetime.now().isoformat()
        tokens = get_claude_token_counts()
        if tokens:
            metrics[f"part{part}_solved_tokens"] = tokens
        save_metrics(day, metrics, year)


def record_execution_time(day: int, part: int, seconds: float, year: int = CURRENT_YEAR):
    """Record execution time for a part."""
    metrics = get_metrics(day, year)
    key = f"part{part}_exec_seconds"
    metrics[key] = seconds
    save_metrics(day, metrics, year)


def get_solve_time(day: int, part: int, year: int = CURRENT_YEAR) -> float | None:
    """Get wallclock time to solve a part in seconds."""
    metrics = get_metrics(day, year)
    started = metrics.get("started_at")
    solved = metrics.get(f"part{part}_solved_at")
    if not started or not solved:
        return None
    start_dt = datetime.fromisoformat(started)
    solved_dt = datetime.fromisoformat(solved)
    return (solved_dt - start_dt).total_seconds()


def get_token_usage(day: int, part: int, year: int = CURRENT_YEAR) -> dict | None:
    """Get token usage to solve a part (delta from start to solve)."""
    metrics = get_metrics(day, year)
    start_tokens = metrics.get("started_tokens")
    end_tokens = metrics.get(f"part{part}_solved_tokens")
    if not start_tokens or not end_tokens:
        return None
    return {
        "input": end_tokens["input"] - start_tokens["input"],
        "output": end_tokens["output"] - start_tokens["output"],
        "cache_read": end_tokens["cache_read"] - start_tokens["cache_read"],
        "cache_create": end_tokens["cache_create"] - start_tokens["cache_create"],
    }


def get_runs_file(year: int = CURRENT_YEAR) -> Path:
    """Get the runs file path."""
    return CACHE_DIR / str(year) / "runs.json"


def get_run_dir(run_id: str, year: int = CURRENT_YEAR) -> Path:
    """Get the directory for a specific run's artifacts."""
    return CACHE_DIR / str(year) / "runs" / run_id


def save_run_artifact(run_id: str, filename: str, content: str, year: int = CURRENT_YEAR):
    """Save an artifact for a run (e.g., solution code, output)."""
    run_dir = get_run_dir(run_id, year)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(content)


def get_run_artifact(run_id: str, filename: str, year: int = CURRENT_YEAR) -> str | None:
    """Get an artifact for a run."""
    artifact_path = get_run_dir(run_id, year) / filename
    if artifact_path.exists():
        return artifact_path.read_text()
    return None


def get_all_runs(year: int = CURRENT_YEAR) -> list[dict]:
    """Get all runs."""
    runs_file = get_runs_file(year)
    if runs_file.exists():
        return json.loads(runs_file.read_text())
    return []


def save_runs(runs: list[dict], year: int = CURRENT_YEAR):
    """Save all runs."""
    runs_file = get_runs_file(year)
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    runs_file.write_text(json.dumps(runs, indent=2))


def get_active_run(day: int, year: int = CURRENT_YEAR) -> dict | None:
    """Get the active (not ended) run for a day, if any."""
    runs = get_all_runs(year)
    for run in reversed(runs):  # Most recent first
        if run["day"] == day and run.get("ended_at") is None:
            return run
    return None


def start_run(day: int, year: int = CURRENT_YEAR, label: str = None) -> dict:
    """Start a new run for a day.

    Returns the new run record.
    """
    import uuid

    model = get_current_model() or "unknown"
    tokens = get_claude_token_counts()

    run = {
        "id": str(uuid.uuid4())[:8],
        "day": day,
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "model": model,
        "label": label or model,
        "start_tokens": tokens,
        "end_tokens": None,
        "part1_solved": False,
        "part2_solved": False,
        "part1_time_ms": None,
        "part2_time_ms": None,
    }

    runs = get_all_runs(year)
    runs.append(run)
    save_runs(runs, year)

    return run


def end_run(day: int, year: int = CURRENT_YEAR, label: str = None) -> dict | None:
    """End the active run for a day.

    Returns the ended run, or None if no active run.
    """
    runs = get_all_runs(year)

    # Find active run for this day
    for run in reversed(runs):
        if run["day"] == day and run.get("ended_at") is None:
            run["ended_at"] = datetime.now().isoformat()
            run["end_tokens"] = get_claude_token_counts()

            # Update label if provided
            if label:
                run["label"] = label

            # Check current metrics for solve status
            metrics = get_metrics(day, year)
            if metrics.get("part1_solved_at"):
                run["part1_solved"] = True
            if metrics.get("part2_solved_at"):
                run["part2_solved"] = True
            if metrics.get("part1_exec_seconds"):
                run["part1_time_ms"] = metrics["part1_exec_seconds"] * 1000
            if metrics.get("part2_exec_seconds"):
                run["part2_time_ms"] = metrics["part2_exec_seconds"] * 1000

            # Also check submissions file (more reliable for "Already completed" cases)
            submissions_file = CACHE_DIR / str(year) / str(day) / "submissions.json"
            if submissions_file.exists():
                try:
                    subs = json.loads(submissions_file.read_text())
                    if any("CORRECT" in s.get("result", "") or "Already completed" in s.get("result", "")
                           for s in subs.get("1", [])):
                        run["part1_solved"] = True
                    if any("CORRECT" in s.get("result", "") or "Already completed" in s.get("result", "")
                           for s in subs.get("2", [])):
                        run["part2_solved"] = True
                except (json.JSONDecodeError, KeyError):
                    pass

            save_runs(runs, year)
            return run

    return None


def get_run_tokens(run: dict) -> dict | None:
    """Calculate token usage for a run (delta between start and end)."""
    start = run.get("start_tokens")
    end = run.get("end_tokens")
    if not start or not end:
        return None
    return {
        "input": end["input"] - start["input"],
        "output": end["output"] - start["output"],
        "cache_read": end["cache_read"] - start["cache_read"],
        "cache_create": end["cache_create"] - start["cache_create"],
    }


def get_day_runs(day: int, year: int = CURRENT_YEAR) -> list[dict]:
    """Get all runs for a specific day."""
    runs = get_all_runs(year)
    return [r for r in runs if r["day"] == day]


def submit(day: int, part: int, answer: str, year: int = CURRENT_YEAR) -> str:
    """Submit an answer for a puzzle part."""
    part_str = str(part)
    answer_str = str(answer).strip()
    timestamp = datetime.now().isoformat()

    submissions = get_submissions(day, year)

    # Check if already submitted
    for attempt in submissions.get(part_str, []):
        if attempt.get("answer") == answer_str:
            cached_result = attempt.get("result")
            print(f"Already submitted this answer: {cached_result}")
            return cached_result

    url = f"{BASE_URL}/{year}/day/{day}/answer"
    response = requests.post(
        url,
        cookies={"session": get_session()},
        data={"level": part, "answer": answer_str},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article")

    if not article:
        result = "Unknown response"
    else:
        text = article.get_text()
        if "That's the right answer" in text:
            result = "CORRECT!"
            record_solved(day, part, year)
            input_content = read_input(day, year)
            save_answer(day, part, answer_str, input_content, year)
            if part == 1:
                print("Fetching updated problem with part 2...")
                fetch_problem(day, year, force=True)
        elif "That's not the right answer" in text:
            if "too high" in text:
                result = "WRONG (too high)"
            elif "too low" in text:
                result = "WRONG (too low)"
            else:
                result = "WRONG"
        elif "You gave an answer too recently" in text:
            result = "RATE LIMITED - wait before trying again"
        elif "already complete" in text.lower():
            result = "Already completed"
        else:
            result = text[:200]

    # Store as list of attempts with timestamps
    if part_str not in submissions:
        submissions[part_str] = []
    submissions[part_str].append({
        "answer": answer_str,
        "result": result,
        "timestamp": timestamp,
    })
    save_submissions(day, submissions, year)

    print(f"Result: {result}")
    return result
