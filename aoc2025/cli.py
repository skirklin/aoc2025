"""CLI for Advent of Code 2025."""

import importlib
import time

import cyclopts

from aoc2025 import (
    CURRENT_YEAR,
    fetch,
    get_input_by_hash,
    get_solve_time,
    get_token_usage,
    list_inputs,
    read_input,
    record_execution_time,
    show,
    submit,
)
from aoc2025 import db

app = cyclopts.App(name="aoc", help="Advent of Code 2025 helper")


def load_day(day: int):
    """Load a day's solution module."""
    module_name = f"aoc2025.days.day{day:02d}"
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


@app.command
def run(day: int, year: int = CURRENT_YEAR):
    """Run solution for a day."""
    module = load_day(day)
    if not module:
        print(f"No solution found for day {day}")
        return

    data = read_input(day, year)

    if hasattr(module, 'part1'):
        start = time.perf_counter()
        result = module.part1(data)
        elapsed = time.perf_counter() - start
        record_execution_time(day, 1, elapsed, year)
        print(f"Part 1: {result}  ({elapsed*1000:.2f}ms)")

    if hasattr(module, 'part2'):
        start = time.perf_counter()
        result = module.part2(data)
        elapsed = time.perf_counter() - start
        record_execution_time(day, 2, elapsed, year)
        print(f"Part 2: {result}  ({elapsed*1000:.2f}ms)")


@app.command
def fetch_day(day: int, year: int = CURRENT_YEAR, *, force: bool = False):
    """Fetch problem and input for a day."""
    problem, input_text = fetch(day, year, force)
    print(f"\n{'='*60}")
    print(f"Day {day} - Problem fetched ({len(problem)} chars)")
    print(f"Input: {len(input_text.splitlines())} lines")
    print(f"{'='*60}\n")
    print(problem)


@app.command
def submit_answer(day: int, part: int, answer: str, year: int = CURRENT_YEAR):
    """Submit an answer for a day."""
    submit(day, part, answer, year)


@app.command
def show_problem(day: int, year: int = CURRENT_YEAR):
    """Show cached problem statement."""
    problem = show(day, year)
    if problem:
        print(problem)


@app.command
def test(day: int, year: int = CURRENT_YEAR):
    """Test solution against examples from problem and cached input/answer pairs."""
    from aoc2025.core import extract_examples

    module = load_day(day)
    if not module:
        print(f"No solution found for day {day}")
        return

    passed = 0
    failed = 0

    # First, test against examples extracted from problem statement
    examples = extract_examples(day, year)
    if examples:
        print("Testing against problem examples:")
        for i, example in enumerate(examples):
            data = example["input"]
            for part_num, part_fn in [("1", "part1"), ("2", "part2")]:
                expected_key = f"part{part_num}"
                expected_val = example.get(expected_key)
                if not expected_val:
                    continue
                if not hasattr(module, part_fn):
                    continue

                try:
                    result = str(getattr(module, part_fn)(data))
                    if result == expected_val:
                        print(f"  [example {i+1}] Part {part_num}: PASS")
                        passed += 1
                    else:
                        print(f"  [example {i+1}] Part {part_num}: FAIL (got {result}, expected {expected_val})")
                        failed += 1
                except Exception as e:
                    print(f"  [example {i+1}] Part {part_num}: ERROR ({e})")
                    failed += 1
    else:
        print("No examples found in problem statement")

    # Then test against cached answers
    answers = db.get_answers(day, year)
    inputs = list_inputs(day, year)

    if answers and inputs:
        print("\nTesting against cached answers:")
        for input_hash in inputs:
            expected = answers.get(input_hash, {})
            if not expected:
                continue

            data = get_input_by_hash(day, input_hash, year)

            for part_num, part_fn in [("1", "part1"), ("2", "part2")]:
                if part_num not in expected:
                    continue
                if not hasattr(module, part_fn):
                    print(f"  [{input_hash[:8]}] Part {part_num}: no {part_fn}() function")
                    failed += 1
                    continue

                result = str(getattr(module, part_fn)(data))
                expected_val = expected[part_num]

                if result == expected_val:
                    print(f"  [{input_hash[:8]}] Part {part_num}: PASS")
                    passed += 1
                else:
                    print(f"  [{input_hash[:8]}] Part {part_num}: FAIL (got {result}, expected {expected_val})")
                    failed += 1

    print(f"\nDay {day}: {passed} passed, {failed} failed")
    return failed == 0


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


@app.command
def stats(day: int, year: int = CURRENT_YEAR):
    """Show performance metrics for a day."""
    metrics = db.get_all_metrics(day, year)
    if not metrics:
        print(f"No metrics for day {day}")
        return

    print(f"Day {day} Metrics:")
    print("-" * 40)

    if "started_at" in metrics:
        print(f"Started: {metrics['started_at']}")

    for part in [1, 2]:
        solved_key = f"part{part}_solved_at"
        exec_key = f"part{part}_exec_seconds"

        if solved_key in metrics:
            print(f"\nPart {part}:")
            print(f"  Solved at: {metrics[solved_key]}")

            solve_time = get_solve_time(day, part, year)
            if solve_time:
                print(f"  Wallclock time to solve: {format_duration(solve_time)}")

            if exec_key in metrics:
                exec_ms = metrics[exec_key] * 1000
                print(f"  Code execution time: {exec_ms:.2f}ms")

            tokens = get_token_usage(day, part, year)
            if tokens:
                total = tokens["input"] + tokens["output"]
                print(f"  Tokens used: {total:,} (in: {tokens['input']:,}, out: {tokens['output']:,})")


@app.command
def start(day: int, year: int = CURRENT_YEAR, *, label: str = None):
    """Start a timed run for a day.

    Use --label to tag this run (e.g., --label "opus-4.5").
    """
    from aoc2025.core import get_claude_token_counts, get_current_model

    # Check for existing active run
    active = db.get_active_run(day, year)
    if active:
        print(f"Day {day} already has an active run (id: {active['id']})")
        print(f"  Started: {active['started_at']}")
        print(f"  Label: {active['label']}")
        print("\nUse 'end' to finish it first, or 'runs' to see all runs.")
        return

    model = get_current_model() or "unknown"
    tokens = get_claude_token_counts()
    run = db.create_run(day, model, label or model, tokens, year)
    print(f"Started run for day {day}")
    print(f"  ID: {run['id']}")
    print(f"  Model: {run['model']}")
    print(f"  Label: {run['label']}")
    print(f"  Started: {run['started_at']}")


@app.command
def end(day: int, year: int = CURRENT_YEAR, *, label: str = None):
    """End the active run for a day.

    Use --label to update the run's label.
    """
    from aoc2025.core import get_claude_token_counts

    active = db.get_active_run(day, year)
    if not active:
        print(f"No active run for day {day}")
        return

    tokens = get_claude_token_counts()
    run = db.end_run(active["id"], tokens, year)

    if label:
        db.update_run(run["id"], year, label=label)
        run["label"] = label

    # Check solve status from submissions
    run["part1_solved"] = db.is_part_solved(day, 1, year)
    run["part2_solved"] = db.is_part_solved(day, 2, year)
    db.update_run(run["id"], year, part1_solved=run["part1_solved"], part2_solved=run["part2_solved"])

    print(f"Ended run for day {day}")
    print(f"  ID: {run['id']}")
    print(f"  Label: {run['label']}")
    print(f"  Started: {run['started_at']}")
    print(f"  Ended: {run['ended_at']}")
    print(f"  Part 1: {'Solved' if run['part1_solved'] else 'Not solved'}", end="")
    if run.get('part1_exec_ms'):
        print(f" ({run['part1_exec_ms']:.2f}ms)")
    else:
        print()
    print(f"  Part 2: {'Solved' if run['part2_solved'] else 'Not solved'}", end="")
    if run.get('part2_exec_ms'):
        print(f" ({run['part2_exec_ms']:.2f}ms)")
    else:
        print()

    tokens = db.get_run_tokens(run)
    if tokens:
        total = tokens['input'] + tokens['output']
        print(f"  Tokens: {total:,} (in: {tokens['input']:,}, out: {tokens['output']:,})")


@app.command
def runs(day: int = None, year: int = CURRENT_YEAR):
    """Show all runs. Optionally filter by day."""
    if day:
        all_runs = db.get_day_runs(day, year)
        if not all_runs:
            print(f"No runs for day {day}")
            return
        print(f"Runs for Day {day}:")
    else:
        all_runs = db.get_all_runs(year)
        if not all_runs:
            print("No runs recorded")
            return
        print(f"All Runs ({year}):")

    print("-" * 95)
    print(f"{'ID':<10} {'Day':<5} {'Label':<20} {'Time':<10} {'P1':<5} {'P2':<5} {'Tokens':<12}")
    print("-" * 95)

    for run in all_runs:
        rid = run["id"]
        d = run["day"]
        label = (run.get("label") or "")[:18]

        # Wall clock time
        wall_secs = run.get("wall_clock_seconds")
        if wall_secs:
            time_str = format_duration(wall_secs)
        elif run.get("ended_at") is None:
            time_str = "running"
        else:
            time_str = "-"

        p1 = "✓" if run.get("part1_solved") else "-"
        p2 = "✓" if run.get("part2_solved") else "-"

        tokens = db.get_run_tokens(run)
        if tokens:
            tok_str = f"{tokens['input'] + tokens['output']:,}"
        else:
            tok_str = "-"

        print(f"{rid:<10} {d:<5} {label:<20} {time_str:<10} {p1:<5} {p2:<5} {tok_str:<12}")


@app.command
def active(year: int = CURRENT_YEAR):
    """Show all active (not ended) runs."""
    all_runs = db.get_all_runs(year)
    active_runs = [r for r in all_runs if r.get("ended_at") is None]

    if not active_runs:
        print("No active runs")
        return

    print("Active Runs:")
    print("-" * 60)
    for run in active_runs:
        print(f"  Day {run['day']}: {run['label']} (id: {run['id']})")
        print(f"    Started: {run['started_at']}")


@app.command
def show_run(run_id: str, year: int = CURRENT_YEAR, *, output: bool = False, code: bool = False):
    """Show details for a specific run.

    Use --output to show Claude's output.
    Use --code to show the solution code.
    """
    run = db.get_run(run_id, year)
    if not run:
        print(f"Run '{run_id}' not found")
        return

    print(f"Run {run['id']}")
    print("-" * 60)
    print(f"  Day: {run['day']}")
    print(f"  Label: {run.get('label', '-')}")
    print(f"  Model: {run.get('model', '-')}")
    print(f"  Started: {run['started_at']}")
    print(f"  Ended: {run.get('ended_at', 'still running')}")

    if run.get('wall_clock_seconds'):
        print(f"  Wall clock: {format_duration(run['wall_clock_seconds'])}")

    print(f"  Part 1: {'✓ Correct' if run.get('part1_solved') else '✗'}", end="")
    if run.get('part1_exec_ms'):
        print(f" (exec: {run['part1_exec_ms']:.2f}ms)")
    else:
        print()

    print(f"  Part 2: {'✓ Correct' if run.get('part2_solved') else '✗'}", end="")
    if run.get('part2_exec_ms'):
        print(f" (exec: {run['part2_exec_ms']:.2f}ms)")
    else:
        print()

    tokens = db.get_run_tokens(run)
    if tokens:
        total = tokens['input'] + tokens['output']
        print(f"  Tokens: {total:,} (in: {tokens['input']:,}, out: {tokens['output']:,})")

    # Show workspace location
    run_dir = db.get_run_dir(run['id'], year)
    workspace = run_dir / "workspace"
    if workspace.exists():
        print(f"\n  Workspace: {workspace}")

    if output:
        out_text = db.get_run_artifact(run['id'], "claude_output.txt", year)
        if out_text:
            print(f"\n{'='*60}")
            print("Claude Output:")
            print("="*60)
            print(out_text)
        else:
            print("\n  (No output artifact saved)")

    if code:
        # Read from workspace
        solution_file = workspace / "solutions" / f"day{run['day']:02d}.py"
        if solution_file.exists():
            print(f"\n{'='*60}")
            print("Solution Code:")
            print("="*60)
            print(solution_file.read_text())
        else:
            print("\n  (No solution file in workspace)")


def setup_benchmark_workspace(run_id: str, day: int, year: int = CURRENT_YEAR) -> "Path":
    """Create an isolated workspace for a benchmark run."""
    from pathlib import Path
    import shutil

    # Create workspace directory
    workspace = db.get_run_dir(run_id, year) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Create solution directory
    days_dir = workspace / "solutions"
    days_dir.mkdir(exist_ok=True)

    # Create empty solution template
    solution_file = days_dir / f"day{day:02d}.py"
    solution_file.write_text(f'''"""Solution for Advent of Code {year} Day {day}."""


def part1(data: str):
    """Solve part 1."""
    pass


def part2(data: str):
    """Solve part 2."""
    pass
''')

    # Copy input file
    input_data = read_input(day, year)
    input_file = workspace / "input.txt"
    input_file.write_text(input_data)

    # Copy problem statement
    problem = show(day, year)
    if problem:
        problem_file = workspace / "problem.md"
        problem_file.write_text(problem)

    # Create a simple runner script
    runner = workspace / "run.py"
    runner.write_text(f'''#!/usr/bin/env python3
"""Runner for day {day} solution."""
import sys
import time
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent))

from solutions.day{day:02d} import part1, part2

def main():
    input_data = Path("input.txt").read_text()

    print("Running solution...")

    start = time.perf_counter()
    result1 = part1(input_data)
    elapsed1 = time.perf_counter() - start
    print(f"Part 1: {{result1}}  ({{elapsed1*1000:.2f}}ms)")

    start = time.perf_counter()
    result2 = part2(input_data)
    elapsed2 = time.perf_counter() - start
    print(f"Part 2: {{result2}}  ({{elapsed2*1000:.2f}}ms)")

if __name__ == "__main__":
    main()
''')

    return workspace


@app.command
def benchmark(
    day: int,
    model: str = "sonnet",
    year: int = CURRENT_YEAR,
    *,
    label: str = None,
    timeout: int = 600,
):
    """Run an automated benchmark with a specific model.

    Spawns Claude Code as a subprocess with the specified model to solve a day.
    Each run gets its own isolated workspace.

    Models: opus, sonnet, haiku (or full model IDs)

    Example:
        python -m aoc2025 benchmark 7 --model sonnet
        python -m aoc2025 benchmark 7 --model opus --label "opus-retry"
    """
    import subprocess
    import os
    from pathlib import Path
    from aoc2025.core import get_claude_token_counts

    # Get problem and input
    problem = show(day, year)
    if not problem:
        print(f"No problem found for day {day}. Run 'fetch-day {day}' first.")
        return

    input_data = read_input(day, year)

    # Create run record first to get ID
    run_label = label or model
    start_tokens = get_claude_token_counts()
    run = db.create_run(day, model, run_label, start_tokens, year)

    # Set up isolated workspace
    workspace = setup_benchmark_workspace(run["id"], day, year)
    print(f"Starting benchmark for day {day} with model '{model}'")
    print(f"  Run ID: {run['id']}")
    print(f"  Label: {run_label}")
    print(f"  Workspace: {workspace}")
    print("-" * 60)

    # Build the prompt
    prompt = f"""You are solving Advent of Code {year} Day {day}. This is an automated benchmark - complete all steps without asking questions.

The problem statement is in problem.md and the input is in input.txt.

Instructions:
1. Read problem.md to understand the problem
2. Write your solution in solutions/day{day:02d}.py - implement part1(data) and part2(data) functions
3. Test by running: python run.py
4. Make sure both parts return the correct answers

The solution file already has a template. Edit it to implement your solution.
"""

    # Spawn Claude Code in the isolated workspace
    cmd = [
        "claude",
        "--model", model,
        "--print",
        "--dangerously-skip-permissions",
        prompt
    ]

    env = os.environ.copy()
    env["CLAUDE_MODEL"] = model

    wall_clock_start = time.perf_counter()
    timed_out = False
    claude_output = ""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        claude_output = result.stdout + (result.stderr or "")
        print(claude_output)
        print("-" * 60)
        print(f"Claude Code exited with code {result.returncode}")
    except subprocess.TimeoutExpired as e:
        claude_output = (e.stdout or b"").decode() + (e.stderr or b"").decode()
        print(claude_output)
        print("-" * 60)
        print(f"Benchmark timed out after {timeout} seconds")
        timed_out = True
    except FileNotFoundError:
        print("Error: 'claude' command not found. Is Claude Code installed?")
        return

    wall_clock_seconds = time.perf_counter() - wall_clock_start

    # Save Claude's output
    db.save_run_artifact(run["id"], "claude_output.txt", claude_output, year)

    # Verify solution against cached answers
    print("\nVerifying solution against cached answers...")

    # Load the solution from the workspace
    solution_file = workspace / "solutions" / f"day{day:02d}.py"
    module = None
    if solution_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"day{day:02d}", solution_file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"  Error loading solution: {e}")
            module = None

    answers = db.get_answers(day, year)

    # Get input hash for answer lookup
    import hashlib
    input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]
    expected = answers.get(input_hash, {})

    part1_solved = False
    part2_solved = False
    part1_exec_ms = None
    part2_exec_ms = None

    if module and expected:
        if hasattr(module, 'part1') and '1' in expected:
            start = time.perf_counter()
            try:
                result = str(module.part1(input_data))
                elapsed = time.perf_counter() - start
                part1_exec_ms = elapsed * 1000
                part1_solved = (result == expected['1'])
                print(f"  Part 1: {result} {'✓' if part1_solved else '✗'} (expected {expected['1']}) [exec: {part1_exec_ms:.2f}ms]")
            except Exception as e:
                print(f"  Part 1: ERROR - {e}")

        if hasattr(module, 'part2') and '2' in expected:
            start = time.perf_counter()
            try:
                result = str(module.part2(input_data))
                elapsed = time.perf_counter() - start
                part2_exec_ms = elapsed * 1000
                part2_solved = (result == expected['2'])
                print(f"  Part 2: {result} {'✓' if part2_solved else '✗'} (expected {expected['2']}) [exec: {part2_exec_ms:.2f}ms]")
            except Exception as e:
                print(f"  Part 2: ERROR - {e}")
    elif not module:
        print(f"  No solution module found for day {day}")
    elif not expected:
        print(f"  No cached answers found for day {day}")

    # End the run with verification results
    end_tokens = get_claude_token_counts()
    db.end_run(run["id"], end_tokens, year)
    db.update_run(
        run["id"], year,
        part1_solved=part1_solved,
        part2_solved=part2_solved,
        wall_clock_seconds=wall_clock_seconds,
        timed_out=timed_out,
        part1_exec_ms=part1_exec_ms,
        part2_exec_ms=part2_exec_ms,
    )

    ended_run = db.get_run(run["id"], year)

    print(f"\nBenchmark complete:")
    print(f"  Wall clock time: {format_duration(wall_clock_seconds)}")
    print(f"  Part 1: {'✓ Correct' if part1_solved else '✗ Wrong/Missing'}", end="")
    if part1_exec_ms:
        print(f" (exec: {part1_exec_ms:.2f}ms)")
    else:
        print()
    print(f"  Part 2: {'✓ Correct' if part2_solved else '✗ Wrong/Missing'}", end="")
    if part2_exec_ms:
        print(f" (exec: {part2_exec_ms:.2f}ms)")
    else:
        print()

    tokens = db.get_run_tokens(ended_run)
    if tokens:
        total = tokens['input'] + tokens['output']
        print(f"  Tokens: {total:,} (in: {tokens['input']:,}, out: {tokens['output']:,})")


@app.command
def tokens():
    """Show current session token usage."""
    from aoc2025.core import get_claude_token_counts

    counts = get_claude_token_counts()
    if not counts:
        print("No token data available")
        return

    print("Current Session Token Usage:")
    print("-" * 35)
    print(f"  Input tokens:        {counts['input']:>12,}")
    print(f"  Output tokens:       {counts['output']:>12,}")
    print(f"  Cache read:          {counts['cache_read']:>12,}")
    print(f"  Cache creation:      {counts['cache_create']:>12,}")
    print("-" * 35)
    print(f"  Total (in + out):    {counts['input'] + counts['output']:>12,}")


@app.command
def dashboard(port: int = None, debug: bool = False):
    """Launch the web dashboard to explore results. Auto-finds free port if not specified."""
    from aoc2025.dashboard import run_dashboard
    print("Press Ctrl+C to stop")
    run_dashboard(port=port, debug=debug)


@app.command
def progress(year: int = CURRENT_YEAR):
    """Show progress summary for all days."""
    from aoc2025.core import CACHE_DIR

    print(f"{'='*50}")
    print(f"  Advent of Code {year} Progress")
    print(f"{'='*50}\n")

    total_stars = 0
    days_data = []

    for day in range(1, 26):
        cache_dir = CACHE_DIR / str(year) / str(day)
        problem_file = cache_dir / "problem.md"

        has_problem = problem_file.exists()
        part1_correct = db.is_part_solved(day, 1, year)
        part2_correct = db.is_part_solved(day, 2, year)

        stars = (1 if part1_correct else 0) + (1 if part2_correct else 0)
        total_stars += stars
        days_data.append((day, has_problem, part1_correct, part2_correct, stars))

    # Display in a compact format
    print("Day   Part 1   Part 2   Stars")
    print("-" * 30)

    for day, has_problem, p1, p2, stars in days_data:
        if not has_problem:
            continue  # Skip days not fetched
        p1_mark = "*" if p1 else "."
        p2_mark = "*" if p2 else "."
        star_display = "*" * stars if stars else ""
        print(f"{day:3d}     {p1_mark}        {p2_mark}      {star_display}")

    print(f"\n{'='*50}")
    print(f"  Total Stars: {total_stars}")
    print(f"{'='*50}")


@app.command
def migrate(year: int = CURRENT_YEAR):
    """Migrate data from JSON files to SQLite database."""
    db.migrate_from_json(year)
    print(f"\nMigration complete. Database: {db.get_db_path(year)}")


@app.command
def freeze(output_dir: str = None):
    """Freeze the dashboard to static HTML for hosting."""
    from pathlib import Path
    from aoc2025.freeze import freeze as do_freeze

    output = Path(output_dir) if output_dir else None
    do_freeze(output)


@app.command
def sync(year: int = CURRENT_YEAR):
    """Fetch all available problems and generate answers from reference solutions."""
    from aoc2025.freeze import ensure_problems_fetched, ensure_answers_generated

    print("Fetching missing problems...")
    fetched = ensure_problems_fetched(year)
    if fetched:
        print(f"  Fetched {len(fetched)} new problems")
    else:
        print("  All problems up to date")

    print("\nGenerating missing answers from reference solutions...")
    generated = ensure_answers_generated(year)
    if generated:
        print(f"  Generated {generated} answers")
    else:
        print("  All answers up to date")

    print("\nSync complete!")


def main():
    app()
