# AoC Benchmark Tool - Design Document

## Purpose

Compare how different AI models (Haiku, Sonnet, Opus) perform on Advent of Code problems. Track correctness, time, and token usage.

## Core Entities

1. **Problem** - The puzzle description from adventofcode.com
2. **Input** - Your personal puzzle input (unique per AoC account)
3. **Reference Solution** - Your correct solution that defines what "right" looks like
4. **Benchmark Run** - A record of an AI model attempting the problem
5. **Expected Answer** - The correct answer for your input (derived from running your solution)

## User Workflows

### Setting Up a New Day

When day N is released on AoC:

1. Download the problem and your input
2. Write your solution in `days/dayNN.py`
3. Run your solution to verify it works and record the correct answers

### Running Benchmarks

1. Pick a day and model
2. Run the benchmark (AI gets isolated workspace with problem + input)
3. Compare AI's answer against your known-correct answer
4. Record results (time, tokens, pass/fail)

### Viewing Results

- View locally via web dashboard
- Export static site to share publicly

## Proposed Commands

### `aoc fetch <day>`
Download problem description and input from adventofcode.com.

- Requires: AOC_SESSION cookie
- Creates: `.cache/YYYY/DD/problem.md`, `.cache/YYYY/DD/inputs/<hash>`

### `aoc test <day>`
Run your reference solution and verify/record the correct answers.

- Requires: `days/dayNN.py` with `part1()` and `part2()` functions
- Creates: Correct answers in database
- Shows: Your solution's output and execution time

This is how the system learns what the "right" answer is for your input.

### `aoc benchmark <day> --model <model>`
Run an AI model on the problem in an isolated workspace.

- Requires: Problem fetched, answers recorded (from `test`)
- Creates: Benchmark run record with timing, tokens, correctness
- Models: `haiku`, `sonnet`, `opus`

### `aoc serve`
Start local web dashboard to browse results.

### `aoc build`
Export dashboard to static HTML in `dist/` for hosting.

### `aoc status`
Show overview: which days are fetched, tested, benchmarked.

```
Day   Problem   Tested   Haiku   Sonnet   Opus
  1      ✓        ✓       **      **       -
  2      ✓        ✓       **      *        -
  3      ✓        -        -       -       -
  ...
```

## Key Principles

### Explicit over implicit
- Fetching requires explicit `fetch` command (network operation)
- Testing requires explicit `test` command (runs your code)
- No magic "sync" that does multiple unrelated things

### Answers come from your solutions
- You can only benchmark days you've solved yourself
- The reference solution IS the source of truth
- `test` runs your solution and caches the answer

### Separation of concerns
- `fetch` = get external data (network)
- `test` = run your code (local)
- `benchmark` = run AI code (subprocess)
- `serve` = view results (local server)
- `build` = export results (file generation)

### Idempotent operations
- `fetch` skips if already fetched
- `test` re-runs solution but updates cached answer if changed
- `benchmark` always creates a new run (intentionally not idempotent)

## What This Replaces

| Old | New | Rationale |
|-----|-----|-----------|
| `run` | `test` | "test" implies verification, not just execution |
| `sync` | (removed) | Split into `fetch --all` + `test --all` |
| `freeze` | `build` | Plain English, matches common tooling (npm, make) |
| `dashboard` | `serve` | Shorter, matches common tooling |

## File Structure

```
aoc2025/
├── days/                    # Your reference solutions
│   ├── day01.py
│   └── ...
├── .cache/
│   └── 2025/
│       ├── aoc.db           # SQLite: runs, answers, metrics
│       ├── 1/
│       │   ├── problem.md   # Puzzle description
│       │   └── inputs/      # Your input(s) by hash
│       └── runs/
│           └── <run_id>/    # Benchmark artifacts
│               ├── workspace/
│               └── claude_output.txt
└── dist/                    # Static site output
```

## Open Questions

1. **Should `benchmark` auto-run `test` if answers are missing?**
   - Pro: Convenient, less commands to remember
   - Con: Violates "explicit over implicit", hides what's happening
   - Recommendation: No, but give clear error message

2. **Should `build` include any data fetching?**
   - No. Build should be pure transformation of current state.
   - If data is missing, show empty dashboard (which is accurate).

3. **Batch operations?**
   - `fetch --all` fetches all available days
   - `test --all` tests all days with solutions
   - `benchmark --all --model X` benchmarks all tested days
