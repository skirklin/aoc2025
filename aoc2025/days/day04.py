"""Day 4: Printing Department"""


def parse(data: str):
    """Parse input data into a set of roll positions."""
    lines = data.strip().split('\n')
    rolls = set()
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == '@':
                rolls.add((r, c))
    return rolls


def count_neighbors(pos, rolls):
    """Count how many rolls are adjacent to a position."""
    r, c = pos
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            if (r + dr, c + dc) in rolls:
                count += 1
    return count


def get_accessible(rolls):
    """Get all rolls that can be accessed (fewer than 4 neighbors)."""
    return {pos for pos in rolls if count_neighbors(pos, rolls) < 4}


def part1(data: str) -> int:
    """Count rolls accessible by a forklift."""
    rolls = parse(data)
    return len(get_accessible(rolls))


def part2(data: str) -> int:
    """Count total rolls that can be removed iteratively."""
    rolls = parse(data)
    total_removed = 0

    while True:
        accessible = get_accessible(rolls)
        if not accessible:
            break
        total_removed += len(accessible)
        rolls -= accessible

    return total_removed
