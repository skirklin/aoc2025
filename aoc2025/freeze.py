"""Freeze the Flask dashboard to static HTML for hosting."""

import warnings
from pathlib import Path
from datetime import datetime

# Suppress Flask-Frozen warnings about unused endpoints and mime types
warnings.filterwarnings("ignore", module="flask_frozen")

from flask_frozen import Freezer

from aoc2025.dashboard import app, get_days_with_problems
from aoc2025 import db
from aoc2025.core import CURRENT_YEAR, CACHE_DIR
from aoc2025 import fetch


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
