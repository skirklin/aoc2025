"""Freeze the Flask dashboard to static HTML for hosting."""

import warnings
from pathlib import Path
from datetime import datetime

# Suppress Flask-Frozen warnings about unused endpoints and mime types
warnings.filterwarnings("ignore", module="flask_frozen")

from flask_frozen import Freezer

from aoc2025.dashboard import app, get_days_with_problems
from aoc2025 import db
from aoc2025.core import CURRENT_YEAR, CACHE_DIR, list_inputs, get_input_by_hash
from aoc2025 import fetch


def ensure_answers_generated(year: int = CURRENT_YEAR):
    """Generate answers for all inputs using reference solutions."""
    import importlib

    generated = 0
    for day in get_days_with_problems(year):
        # Try to load the reference solution
        try:
            module = importlib.import_module(f"aoc2025.days.day{day:02d}")
        except ImportError:
            continue

        # Get all inputs for this day
        inputs = list_inputs(day, year)
        for input_hash in inputs:
            # Check if we already have answers for this input
            existing = db.get_answers(day, year)
            if input_hash in existing and '1' in existing[input_hash] and '2' in existing[input_hash]:
                continue

            # Run the reference solution
            data = get_input_by_hash(day, input_hash, year)
            if data is None:
                continue

            if hasattr(module, 'part1'):
                try:
                    result = module.part1(data)
                    if result is not None:
                        db.save_answer(day, input_hash, 1, str(result), year)
                        generated += 1
                except Exception:
                    pass

            if hasattr(module, 'part2'):
                try:
                    result = module.part2(data)
                    if result is not None:
                        db.save_answer(day, input_hash, 2, str(result), year)
                        generated += 1
                except Exception:
                    pass

    return generated


def ensure_problems_fetched(year: int = CURRENT_YEAR):
    """Fetch any missing problems that should be available."""
    # Determine which days are available (AoC releases one per day Dec 1-25)
    now = datetime.now()
    if now.year == year and now.month == 12:
        max_day = min(now.day, 25)
    elif now.year > year or (now.year == year and now.month > 12):
        max_day = 25
    else:
        max_day = 0

    fetched = []
    for day in range(1, max_day + 1):
        problem_file = CACHE_DIR / str(year) / str(day) / "problem.md"
        if not problem_file.exists():
            print(f"  Fetching day {day}...")
            try:
                fetch(day, year)
                fetched.append(day)
            except Exception as e:
                print(f"  Warning: Could not fetch day {day}: {e}")

    return fetched


# Add routes with explicit .html extensions for static hosting
@app.route("/day/<int:day>.html")
def day_detail_static(day: int):
    from aoc2025.dashboard import day_detail
    return day_detail(day)


@app.route("/run/<run_id>.html")
def run_detail_static(run_id: str):
    from aoc2025.dashboard import run_detail
    return run_detail(run_id)


@app.route("/api/comparison.json")
def api_comparison_static():
    from aoc2025.dashboard import api_comparison
    return api_comparison()


@app.route("/api/runs.json")
def api_runs_static():
    from aoc2025.dashboard import api_runs
    return api_runs()


@app.route("/api/run/<run_id>.json")
def api_run_static(run_id: str):
    from aoc2025.dashboard import api_run
    return api_run(run_id)


def freeze(output_dir: Path = None):
    """Freeze the Flask app to static HTML."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "dist"

    # Ensure all available problems are fetched
    print("Checking for missing problems...")
    fetched = ensure_problems_fetched()
    if fetched:
        print(f"  Fetched {len(fetched)} new problems")
    else:
        print("  All problems up to date")

    # Ensure answers are generated from reference solutions
    print("\nGenerating missing answers from reference solutions...")
    generated = ensure_answers_generated()
    if generated:
        print(f"  Generated {generated} answers")
    else:
        print("  All answers up to date")

    app.config["FREEZER_DESTINATION"] = str(output_dir)
    app.config["FREEZER_RELATIVE_URLS"] = True

    freezer = Freezer(app)

    @freezer.register_generator
    def url_generator():
        # Index page
        yield "/"

        # Day detail pages with .html extension
        for day in get_days_with_problems():
            yield f"/day/{day}.html"

        # Run detail pages with .html extension
        runs = db.get_all_runs(CURRENT_YEAR)
        for run in runs:
            yield f"/run/{run['id']}.html"

        # API endpoints with .json extension
        yield "/api/comparison.json"
        yield "/api/runs.json"
        for run in runs:
            yield f"/api/run/{run['id']}.json"

    print(f"\nFreezing to {output_dir}...")
    freezer.freeze()
    print(f"Done! Static site in {output_dir}")
    print(f"\nTo preview locally:")
    print(f"  cd {output_dir} && python -m http.server 8000")


if __name__ == "__main__":
    freeze()
