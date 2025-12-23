"""Day 2: Gift Shop"""


def parse(data: str):
    """Parse input data into list of (start, end) ranges."""
    ranges = []
    for part in data.strip().split(','):
        if part.strip():
            start, end = part.strip().split('-')
            ranges.append((int(start), int(end)))
    return ranges


def is_repeated_twice(n: int) -> bool:
    """Check if number is made of a sequence repeated exactly twice."""
    s = str(n)
    length = len(s)
    if length % 2 != 0:
        return False
    half = length // 2
    return s[:half] == s[half:]


def is_repeated_at_least_twice(n: int) -> bool:
    """Check if number is made of a sequence repeated at least twice."""
    s = str(n)
    length = len(s)
    # Try all possible pattern lengths from 1 to length//2
    for pat_len in range(1, length // 2 + 1):
        if length % pat_len == 0:
            pattern = s[:pat_len]
            if pattern * (length // pat_len) == s:
                return True
    return False


def part1(data: str) -> int:
    """Sum all invalid IDs (sequence repeated exactly twice)."""
    ranges = parse(data)
    total = 0
    for start, end in ranges:
        for n in range(start, end + 1):
            if is_repeated_twice(n):
                total += n
    return total


def part2(data: str) -> int:
    """Sum all invalid IDs (sequence repeated at least twice)."""
    ranges = parse(data)
    total = 0
    for start, end in ranges:
        for n in range(start, end + 1):
            if is_repeated_at_least_twice(n):
                total += n
    return total
