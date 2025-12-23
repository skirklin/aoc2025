"""Day 3: Lobby"""


def parse(data: str):
    """Parse input data into list of battery banks."""
    return data.strip().split('\n')


def max_joltage(bank: str, k: int) -> int:
    """Find the maximum k-digit number by selecting k batteries in order."""
    n = len(bank)
    result = []
    start = 0

    for remaining in range(k, 0, -1):
        # We need to pick from positions start to n-remaining (inclusive)
        # to leave room for remaining-1 more picks
        end = n - remaining
        # Find the position of the max digit in range [start, end]
        best_pos = start
        best_digit = bank[start]
        for i in range(start + 1, end + 1):
            if bank[i] > best_digit:
                best_digit = bank[i]
                best_pos = i
        result.append(best_digit)
        start = best_pos + 1

    return int(''.join(result))


def part1(data: str) -> int:
    """Sum of max 2-digit joltages from each bank."""
    banks = parse(data)
    return sum(max_joltage(bank, 2) for bank in banks)


def part2(data: str) -> int:
    """Sum of max 12-digit joltages from each bank."""
    banks = parse(data)
    return sum(max_joltage(bank, 12) for bank in banks)
