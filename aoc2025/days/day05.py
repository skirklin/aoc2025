"""Day 5: Cafeteria"""


def parse(data: str):
    """Parse input into ranges and available IDs."""
    parts = data.strip().split('\n\n')
    ranges = []
    for line in parts[0].split('\n'):
        start, end = line.split('-')
        ranges.append((int(start), int(end)))

    available = []
    if len(parts) > 1:
        for line in parts[1].split('\n'):
            available.append(int(line))

    return ranges, available


def is_fresh(ingredient_id, ranges):
    """Check if an ingredient ID is in any range."""
    for start, end in ranges:
        if start <= ingredient_id <= end:
            return True
    return False


def merge_ranges(ranges):
    """Merge overlapping ranges and return list of non-overlapping ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 1:
            # Overlapping or adjacent, extend
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def part1(data: str) -> int:
    """Count available ingredients that are fresh."""
    ranges, available = parse(data)
    return sum(1 for a in available if is_fresh(a, ranges))


def part2(data: str) -> int:
    """Count total unique fresh ingredient IDs."""
    ranges, _ = parse(data)
    merged = merge_ranges(ranges)
    return sum(end - start + 1 for start, end in merged)
