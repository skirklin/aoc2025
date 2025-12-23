"""Web dashboard for exploring AoC 2025 benchmark results."""

from pathlib import Path

from flask import Flask, render_template, jsonify
import markdown

from aoc2025.core import CACHE_DIR, CURRENT_YEAR, get_input_by_hash, list_inputs
from aoc2025 import db

app = Flask(__name__, template_folder="templates", static_folder="static")


def get_benchmark_summary(year: int = CURRENT_YEAR) -> dict:
    """Get summary of all benchmark runs grouped by model and day."""
    runs = db.get_all_runs(year)

    # Group runs by model
    models = {}
    for run in runs:
        model = run.get("model") or run.get("label") or "unknown"
        if model not in models:
            models[model] = {"runs": [], "days": {}}
        models[model]["runs"].append(run)

        day = run["day"]
        if day not in models[model]["days"]:
            models[model]["days"][day] = []
        models[model]["days"][day].append(run)

    # Calculate stats per model
    for model, data in models.items():
        total_runs = len(data["runs"])
        solved_p1 = sum(1 for r in data["runs"] if r.get("part1_solved"))
        solved_p2 = sum(1 for r in data["runs"] if r.get("part2_solved"))
        total_time = sum(r.get("wall_clock_seconds") or 0 for r in data["runs"])
        total_tokens = sum(
            (db.get_run_tokens(r) or {}).get("input", 0) + (db.get_run_tokens(r) or {}).get("output", 0)
            for r in data["runs"]
        )

        data["stats"] = {
            "total_runs": total_runs,
            "solved_p1": solved_p1,
            "solved_p2": solved_p2,
            "total_time": total_time,
            "total_tokens": total_tokens,
        }

    return models


def get_days_with_problems(year: int = CURRENT_YEAR) -> list[int]:
    """Get list of days that have problems fetched."""
    days = []
    for day in range(1, 26):
        problem_file = CACHE_DIR / str(year) / str(day) / "problem.md"
        if problem_file.exists():
            days.append(day)
    return days


def get_comparison_matrix(year: int = CURRENT_YEAR) -> dict:
    """Build a matrix comparing models across days."""
    runs = db.get_all_runs(year)
    days = get_days_with_problems(year)

    # Get unique models
    models = sorted(set(r.get("model") or r.get("label") or "unknown" for r in runs))

    # Build matrix: day -> model -> best run
    matrix = {}
    for day in days:
        matrix[day] = {}
        for model in models:
            # Get best run for this day/model (most recent successful, or most recent)
            day_model_runs = [
                r for r in runs
                if r["day"] == day and (r.get("model") or r.get("label")) == model
            ]
            if day_model_runs:
                # Prefer solved runs, then by recency
                solved_runs = [r for r in day_model_runs if r.get("part1_solved") or r.get("part2_solved")]
                best = solved_runs[-1] if solved_runs else day_model_runs[-1]
                tokens = db.get_run_tokens(best)
                matrix[day][model] = {
                    "run_id": best["id"],
                    "p1": best.get("part1_solved", False),
                    "p2": best.get("part2_solved", False),
                    "time": best.get("wall_clock_seconds"),
                    "tokens": (tokens["input"] + tokens["output"]) if tokens else None,
                    "p1_exec_ms": best.get("part1_exec_ms"),
                    "p2_exec_ms": best.get("part2_exec_ms"),
                }

    return {"days": days, "models": models, "matrix": matrix}


def get_run_detail(run_id: str, year: int = CURRENT_YEAR) -> dict:
    """Get detailed data for a specific run."""
    run = db.get_run(run_id, year)
    if not run:
        return None

    day = run["day"]
    run_dir = db.get_run_dir(run_id, year)
    workspace = run_dir / "workspace"

    data = {
        "run": run,
        "tokens": db.get_run_tokens(run),
        "problem_html": "",
        "solution_code": "",
        "claude_output": "",
    }

    # Problem statement
    problem_file = workspace / "problem.md"
    if not problem_file.exists():
        problem_file = CACHE_DIR / str(year) / str(day) / "problem.md"
    if problem_file.exists():
        data["problem_html"] = markdown.markdown(
            problem_file.read_text(),
            extensions=["fenced_code", "tables"]
        )

    # Solution code from workspace
    solution_file = workspace / "solutions" / f"day{day:02d}.py"
    if solution_file.exists():
        data["solution_code"] = solution_file.read_text()

    # Claude output
    output_file = run_dir / "claude_output.txt"
    if output_file.exists():
        data["claude_output"] = output_file.read_text()

    return data


def get_day_runs(day: int, year: int = CURRENT_YEAR) -> dict:
    """Get all runs for a specific day."""
    runs = db.get_day_runs(day, year)

    # Problem statement
    problem_file = CACHE_DIR / str(year) / str(day) / "problem.md"
    problem_html = ""
    if problem_file.exists():
        problem_html = markdown.markdown(
            problem_file.read_text(),
            extensions=["fenced_code", "tables"]
        )

    # Enrich runs with token data
    for run in runs:
        run["tokens"] = db.get_run_tokens(run)

    # Expected answers
    answers = db.get_answers(day, year)

    return {
        "day": day,
        "year": year,
        "runs": runs,
        "problem_html": problem_html,
        "answers": answers,
    }


@app.route("/")
def index():
    """Main dashboard page - model comparison matrix."""
    comparison = get_comparison_matrix()
    models_summary = get_benchmark_summary()
    return render_template(
        "index.html",
        comparison=comparison,
        models=models_summary,
        year=CURRENT_YEAR
    )


@app.route("/day/<int:day>")
def day_detail(day: int):
    """Detail page for a specific day - shows all runs."""
    data = get_day_runs(day)
    return render_template("day_detail.html", **data)


@app.route("/run/<run_id>")
def run_detail(run_id: str):
    """Detail page for a specific run."""
    data = get_run_detail(run_id)
    if not data:
        return "Run not found", 404
    return render_template("run_detail.html", **data)


@app.route("/api/comparison")
def api_comparison():
    """API endpoint for comparison matrix."""
    return jsonify(get_comparison_matrix())


@app.route("/api/runs")
def api_runs():
    """API endpoint for all runs."""
    return jsonify(db.get_all_runs())


@app.route("/api/run/<run_id>")
def api_run(run_id: str):
    """API endpoint for run detail."""
    data = get_run_detail(run_id)
    if not data:
        return jsonify({"error": "not found"}), 404
    # Remove non-serializable content
    data.pop("problem_html", None)
    return jsonify(data)


def find_free_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


def run_dashboard(host: str = "127.0.0.1", port: int = None, debug: bool = False):
    """Run the dashboard server. If port is None, finds a free port automatically."""
    if port is None:
        port = find_free_port()
    print(f"Dashboard available at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
