"""Day 7: Laboratories"""
from collections import defaultdict


def parse(data: str):
    """Parse input into grid and find start position."""
    lines = data.strip().split('\n')
    grid = [list(line) for line in lines]
    start = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == 'S':
                start = (r, c)
                break
        if start:
            break
    return grid, start


def simulate(grid, start):
    """Simulate tachyon beams and count splits."""
    rows = len(grid)
    cols = len(grid[0])

    # Track active beams at each column, moving downward
    # Use a set to avoid counting duplicate beams at same position
    # beam_cols[row] = set of columns with active beams
    beam_cols = defaultdict(set)
    beam_cols[start[0]].add(start[1])

    splits = 0

    for r in range(start[0], rows - 1):
        next_beams = set()
        for c in beam_cols[r]:
            # Beam at (r, c) moves down to (r+1, c)
            next_r = r + 1
            if next_r >= rows:
                continue

            ch = grid[next_r][c]
            if ch == '^':
                # Split: beam stops, new beams go left and right
                splits += 1
                if c - 1 >= 0:
                    next_beams.add(c - 1)
                if c + 1 < cols:
                    next_beams.add(c + 1)
            else:
                # Continue downward
                next_beams.add(c)

        beam_cols[next_r] = next_beams

    return splits


def part1(data: str) -> int:
    """Count total beam splits."""
    grid, start = parse(data)
    return simulate(grid, start)


def simulate_timelines(grid, start):
    """Simulate tachyon particle and count timelines."""
    rows = len(grid)
    cols = len(grid[0])

    # Track timeline counts at each column
    # beam_counts[col] = number of timelines at this beam position
    beam_counts = defaultdict(int)
    beam_counts[start[1]] = 1

    for r in range(start[0], rows - 1):
        next_counts = defaultdict(int)
        for c, count in beam_counts.items():
            next_r = r + 1
            if next_r >= rows:
                continue

            ch = grid[next_r][c]
            if ch == '^':
                # Split: each timeline becomes two (left and right)
                if c - 1 >= 0:
                    next_counts[c - 1] += count
                if c + 1 < cols:
                    next_counts[c + 1] += count
            else:
                # Continue downward
                next_counts[c] += count

        beam_counts = next_counts

    # Total timelines is sum of all beam counts
    return sum(beam_counts.values())


def part2(data: str) -> int:
    """Count total timelines after particle passes through manifold."""
    grid, start = parse(data)
    return simulate_timelines(grid, start)
