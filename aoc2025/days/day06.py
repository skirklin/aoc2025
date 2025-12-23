"""Day 6: Trash Compactor"""
from functools import reduce
from operator import add, mul


def parse(data: str):
    """Parse input into grid of characters."""
    lines = data.split('\n')
    # Remove trailing empty line if present
    while lines and not lines[-1]:
        lines.pop()
    # Pad all lines to same length
    max_len = max(len(line) for line in lines)
    grid = [line.ljust(max_len) for line in lines]
    return grid


def find_problems(grid):
    """Find problem boundaries (columns that are all spaces separate problems)."""
    rows = len(grid) - 1  # Exclude operator row
    cols = len(grid[0])

    # Find separator columns (all spaces in data rows)
    separators = []
    for c in range(cols):
        if all(grid[r][c] == ' ' for r in range(rows)):
            separators.append(c)

    # Group consecutive separators and find problem ranges
    problems = []
    start = 0
    for c in range(cols):
        if c in separators:
            if start < c:
                problems.append((start, c - 1))
            start = c + 1
    if start < cols:
        problems.append((start, cols - 1))

    return problems


def solve_part1(grid):
    """Solve using horizontal number reading."""
    problems = find_problems(grid)
    operator_row = grid[-1]
    data_rows = grid[:-1]

    total = 0
    for start, end in problems:
        # Get operator (find non-space in operator row for this problem)
        op = None
        for c in range(start, end + 1):
            if operator_row[c] in '+*':
                op = operator_row[c]
                break

        # Get numbers (each row has one number, read left-to-right)
        numbers = []
        for row in data_rows:
            num_str = row[start:end + 1].strip()
            if num_str:
                numbers.append(int(num_str))

        # Calculate result
        if op == '+':
            result = sum(numbers)
        else:  # *
            result = reduce(mul, numbers, 1)
        total += result

    return total


def solve_part2(grid):
    """Solve using column-based number reading (right-to-left)."""
    problems = find_problems(grid)
    operator_row = grid[-1]
    data_rows = grid[:-1]

    total = 0
    for start, end in problems:
        # Get operator
        op = None
        for c in range(start, end + 1):
            if operator_row[c] in '+*':
                op = operator_row[c]
                break

        # Get numbers: each column is a number, reading right-to-left
        # Within each column, top to bottom is most to least significant
        numbers = []
        for c in range(end, start - 1, -1):
            digits = []
            for row in data_rows:
                if row[c] != ' ':
                    digits.append(row[c])
            if digits:
                numbers.append(int(''.join(digits)))

        # Calculate result
        if op == '+':
            result = sum(numbers)
        else:  # *
            result = reduce(mul, numbers, 1)
        total += result

    return total


def part1(data: str) -> int:
    """Solve with horizontal number reading."""
    grid = parse(data)
    return solve_part1(grid)


def part2(data: str) -> int:
    """Solve with column-based number reading."""
    grid = parse(data)
    return solve_part2(grid)
