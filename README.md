# AoC 2025 Benchmark Harness

A benchmarking harness for comparing AI models (Claude Haiku, Sonnet, Opus) on Advent of Code 2025 problems.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file with your AoC session cookie:
```
AOC_SESSION=your_session_cookie_here
```

## Commands

### `fetch` - Download problem and input
```bash
python -m aoc2025 fetch <day>      # Fetch one day
python -m aoc2025 fetch --all      # Fetch all available days
```

### `test` - Run your solution and cache answers
```bash
python -m aoc2025 test <day>       # Test one day
python -m aoc2025 test --all       # Test all days with solutions
```

This runs your reference solution and saves the answers so benchmarks can verify correctness.

### `benchmark` - Run AI model on a problem
```bash
python -m aoc2025 benchmark <day> --model haiku
python -m aoc2025 benchmark <day> --model sonnet
python -m aoc2025 benchmark <day> --model opus
```

Requires: Problem fetched and answers cached (run `test` first).

### `status` - Show overview
```bash
python -m aoc2025 status
```

Shows which days are fetched, tested, and benchmarked by model.

### `serve` - Start local dashboard
```bash
python -m aoc2025 serve
```

### `build` - Export static HTML
```bash
python -m aoc2025 build
```

Exports to `dist/` folder for hosting.

## Workflow

1. **Fetch** the problem: `python -m aoc2025 fetch 1`
2. **Write** your solution in `aoc2025/days/day01.py`
3. **Test** to cache answers: `python -m aoc2025 test 1`
4. **Benchmark** AI models: `python -m aoc2025 benchmark 1 --model sonnet`
5. **View** results: `python -m aoc2025 serve`

## Running All Benchmarks

```bash
./run_benchmarks.sh                    # All models, all days
./run_benchmarks.sh haiku              # Just haiku
./run_benchmarks.sh "haiku sonnet"     # Multiple models
./run_benchmarks.sh haiku "1 2 3"      # Specific days
```

## Deployment

### Generate static site
```bash
python -m aoc2025 build
```

### Deploy to GitHub Pages
```bash
./deploy.sh
```

### Custom domain
Add CNAME record in your DNS:
- Host: `aoc`
- Value: `<username>.github.io`

## Project Structure

```
aoc2025/
├── aoc2025/
│   ├── cli.py          # Command-line interface
│   ├── core.py         # Core utilities
│   ├── dashboard.py    # Flask web dashboard
│   ├── db.py           # SQLite database
│   ├── freeze.py       # Static site generator
│   └── days/           # Your reference solutions
├── .cache/             # Problem data and benchmark runs (gitignored)
├── dist/               # Generated static site (gitignored)
└── run_benchmarks.sh   # Batch benchmark script
```
