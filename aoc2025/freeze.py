"""Freeze the Flask dashboard to static HTML for hosting."""

import warnings
from pathlib import Path

# Suppress Flask-Frozen warnings about unused endpoints and mime types
warnings.filterwarnings("ignore", module="flask_frozen")

from flask_frozen import Freezer

from aoc2025.dashboard import app, get_days_with_problems
from aoc2025 import db
from aoc2025.core import CURRENT_YEAR


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

    print(f"Freezing to {output_dir}...")
    freezer.freeze()
    print(f"Done! Static site in {output_dir}")
    print(f"\nTo preview locally:")
    print(f"  cd {output_dir} && python -m http.server 8000")


if __name__ == "__main__":
    freeze()
