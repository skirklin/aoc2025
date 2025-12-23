# How to Solve Advent of Code Problems

## Workflow

1. **Fetch the problem**:
   ```bash
   python -m aoc2025 fetch-day <day>
   ```
   This caches the problem statement and input, and **records the start timestamp and token count** for metrics.

2. **Create a solution file** at `aoc2025/days/day<NN>.py` with this structure:
   ```python
   """Day N: <Title>"""

   def parse(data: str):
       """Parse input data."""
       # Return parsed data structure
       pass

   def part1(data: str) -> int:
       """Solve part 1."""
       pass

   def part2(data: str) -> int:
       """Solve part 2."""
       pass
   ```

3. **Run the solution**:
   ```bash
   python -m aoc2025 run <day>
   ```
   This shows results with execution times.

4. **Submit answers**:
   ```bash
   python -m aoc2025 submit-answer <day> <part> <answer>
   ```
   When correct, this **records the solve timestamp and token count**.

5. **View metrics**:
   ```bash
   python -m aoc2025 stats <day>
   ```
   Shows wallclock solve time, code execution time, and token usage.

## Metrics Tracked

- **Wallclock time**: Time from `fetch-day` to correct submission
- **Code execution time**: How long `part1`/`part2` functions take to run
- **Token usage**: Claude tokens used between start and solve (read from `~/.claude/stats-cache.json`)

All metrics stored in `.cache/2025/<day>/metrics.json`.

## Other Commands

- `python -m aoc2025 show-problem <day>` - View cached problem
- `python -m aoc2025 test <day>` - Test against cached input/answer pairs

## File Structure

```
aoc2025/
├── __init__.py      # Public API exports
├── __main__.py      # Entry point
├── cli.py           # CLI commands (cyclopts)
├── core.py          # Fetching, caching, metrics, submission
└── days/
    └── day01.py     # Solutions go here
```

## Cache Structure

```
.cache/2025/<day>/
├── problem.md       # Problem statement
├── input_hash.txt   # Reference to primary input
├── inputs/
│   └── <hash>       # Content-addressable input storage
├── answers.json     # Correct answers by input hash
├── submissions.json # All submission attempts with timestamps
└── metrics.json     # Performance metrics
```
