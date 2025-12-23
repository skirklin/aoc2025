"""Day 1: Secret Entrance"""


def parse(data: str):
    """Parse input data into list of (direction, distance) tuples."""
    rotations = []
    for line in data.strip().split('\n'):
        direction = line[0]  # 'L' or 'R'
        distance = int(line[1:])
        rotations.append((direction, distance))
    return rotations


def part1(data: str) -> int:
    """Count times dial points at 0 after any rotation."""
    rotations = parse(data)
    position = 50
    count = 0

    for direction, distance in rotations:
        if direction == 'L':
            position = (position - distance) % 100
        else:  # R
            position = (position + distance) % 100

        if position == 0:
            count += 1

    return count


def part2(data: str) -> int:
    """Count times any click causes dial to point at 0."""
    rotations = parse(data)
    position = 50
    count = 0

    for direction, distance in rotations:
        for _ in range(distance):
            if direction == 'L':
                position = (position - 1) % 100
            else:  # R
                position = (position + 1) % 100

            if position == 0:
                count += 1

    return count
