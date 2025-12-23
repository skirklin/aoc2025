"""Day 10: Factory"""
import re
from itertools import product


def parse(data: str):
    """Parse input into list of (target, buttons) tuples."""
    machines = []
    for line in data.strip().split('\n'):
        # Extract indicator pattern
        pattern_match = re.search(r'\[([.#]+)\]', line)
        pattern = pattern_match.group(1)
        target = [1 if c == '#' else 0 for c in pattern]

        # Extract button wirings
        buttons = []
        for match in re.finditer(r'\(([0-9,]+)\)', line):
            indices = [int(x) for x in match.group(1).split(',')]
            buttons.append(indices)

        machines.append((target, buttons))
    return machines


def solve_machine(target, buttons):
    """Find minimum button presses to achieve target state."""
    n_lights = len(target)
    n_buttons = len(buttons)

    if n_buttons == 0:
        return 0 if all(t == 0 for t in target) else float('inf')

    # For small number of buttons, brute force all combinations
    if n_buttons <= 20:
        min_presses = float('inf')
        for mask in range(1 << n_buttons):
            state = [0] * n_lights
            presses = 0
            for b in range(n_buttons):
                if mask & (1 << b):
                    presses += 1
                    for idx in buttons[b]:
                        state[idx] ^= 1
            if state == target:
                min_presses = min(min_presses, presses)
        return min_presses

    # For larger cases, use Gaussian elimination over GF(2)
    # Build augmented matrix [A | b]
    matrix = []
    for i in range(n_lights):
        row = [0] * (n_buttons + 1)
        for b, indices in enumerate(buttons):
            if i in indices:
                row[b] = 1
        row[n_buttons] = target[i]
        matrix.append(row)

    # Gaussian elimination
    pivot_row = 0
    pivot_cols = []
    for col in range(n_buttons):
        # Find pivot
        found = False
        for row in range(pivot_row, n_lights):
            if matrix[row][col] == 1:
                matrix[pivot_row], matrix[row] = matrix[row], matrix[pivot_row]
                found = True
                break
        if not found:
            continue
        pivot_cols.append(col)
        # Eliminate
        for row in range(n_lights):
            if row != pivot_row and matrix[row][col] == 1:
                for c in range(n_buttons + 1):
                    matrix[row][c] ^= matrix[pivot_row][c]
        pivot_row += 1

    # Check for inconsistency
    for row in range(pivot_row, n_lights):
        if matrix[row][n_buttons] == 1:
            return float('inf')  # No solution

    # Find minimum weight solution
    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(n_buttons) if c not in pivot_cols]
    n_free = len(free_cols)

    min_presses = float('inf')
    for free_vals in product([0, 1], repeat=n_free):
        solution = [0] * n_buttons
        for i, col in enumerate(free_cols):
            solution[col] = free_vals[i]

        # Back-substitute to find pivot variable values
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            val = matrix[i][n_buttons]
            for c in range(col + 1, n_buttons):
                val ^= matrix[i][c] * solution[c]
            solution[col] = val

        presses = sum(solution)
        min_presses = min(min_presses, presses)

    return min_presses


def part1(data: str) -> int:
    """Sum of minimum button presses for all machines."""
    machines = parse(data)
    total = 0
    for target, buttons in machines:
        total += solve_machine(target, buttons)
    return total


def parse_with_joltage(data: str):
    """Parse input into list of (buttons, joltage) tuples."""
    machines = []
    for line in data.strip().split('\n'):
        # Extract button wirings
        buttons = []
        for match in re.finditer(r'\(([0-9,]+)\)', line):
            indices = [int(x) for x in match.group(1).split(',')]
            buttons.append(indices)

        # Extract joltage requirements
        joltage_match = re.search(r'\{([0-9,]+)\}', line)
        joltage = [int(x) for x in joltage_match.group(1).split(',')]

        machines.append((buttons, joltage))
    return machines


def solve_joltage(buttons, target, timeout_sec=300):
    """Find minimum button presses using Gaussian elimination and parameter search."""
    from fractions import Fraction
    import time

    start_time = time.time()

    n_counters = len(target)
    n_buttons = len(buttons)

    if n_buttons == 0:
        return 0 if all(t == 0 for t in target) else float('inf')

    # Build augmented matrix [A | b] using Fractions for exact arithmetic
    matrix = []
    for i in range(n_counters):
        row = [Fraction(0)] * (n_buttons + 1)
        for j, indices in enumerate(buttons):
            if i in indices:
                row[j] = Fraction(1)
        row[n_buttons] = Fraction(target[i])
        matrix.append(row)

    # Gaussian elimination to row echelon form
    pivot_row = 0
    pivot_cols = []
    for col in range(n_buttons):
        # Find pivot
        found = -1
        for row in range(pivot_row, n_counters):
            if matrix[row][col] != 0:
                found = row
                break
        if found == -1:
            continue

        # Swap rows
        matrix[pivot_row], matrix[found] = matrix[found], matrix[pivot_row]
        pivot_cols.append(col)

        # Scale pivot row
        scale = matrix[pivot_row][col]
        for c in range(n_buttons + 1):
            matrix[pivot_row][c] /= scale

        # Eliminate column
        for row in range(n_counters):
            if row != pivot_row and matrix[row][col] != 0:
                factor = matrix[row][col]
                for c in range(n_buttons + 1):
                    matrix[row][c] -= factor * matrix[pivot_row][c]

        pivot_row += 1

    # Check for inconsistency (0 = nonzero)
    for row in range(pivot_row, n_counters):
        if matrix[row][n_buttons] != 0:
            return float('inf')

    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(n_buttons) if c not in pivot_cols]
    n_free = len(free_cols)

    # For each pivot column, express it in terms of free variables
    # pivot_var = b - sum(coef * free_var)
    pivot_expressions = []  # (constant, {free_col: coef})
    for i, pc in enumerate(pivot_cols):
        const = matrix[i][n_buttons]
        coeffs = {}
        for fc in free_cols:
            if matrix[i][fc] != 0:
                coeffs[fc] = -matrix[i][fc]
        pivot_expressions.append((const, coeffs))

    # If no free variables, just compute the solution directly
    if n_free == 0:
        total = 0
        for i, pc in enumerate(pivot_cols):
            val = matrix[i][n_buttons]
            if val < 0 or val.denominator != 1:
                return float('inf')
            total += int(val)
        return total

    # Simple recursive search with dynamic pruning
    min_total = [float('inf')]
    max_val = max(target) + 1 if target else 1

    def search(idx, free_vals, partial_pivots):
        if time.time() - start_time > timeout_sec:
            return

        # Compute current total contribution from free vars
        free_sum = sum(free_vals)
        if free_sum >= min_total[0]:
            return

        if idx == n_free:
            # Check if all pivots are valid non-negative integers
            total = free_sum
            for pv in partial_pivots:
                if pv < 0 or pv.denominator != 1:
                    return
                total += int(pv)
            min_total[0] = min(min_total[0], total)
            return

        fc = free_cols[idx]

        for v in range(max_val + 1):
            if free_sum + v >= min_total[0]:
                break  # Pruning: can't do better

            # Update partial pivots
            new_pivots = []
            valid = True
            for i, (const, coeffs) in enumerate(pivot_expressions):
                pv = partial_pivots[i]
                if fc in coeffs:
                    pv = pv + coeffs[fc] * v
                new_pivots.append(pv)

            search(idx + 1, free_vals + [v], new_pivots)

    # Initial partial pivots are just the constants
    initial_pivots = [const for const, _ in pivot_expressions]
    search(0, [], initial_pivots)

    return min_total[0]


def part2(data: str) -> int:
    """Sum of minimum button presses for joltage configuration."""
    machines = parse_with_joltage(data)
    total = 0
    for buttons, joltage in machines:
        result = solve_joltage(buttons, joltage, timeout_sec=30)  # 30 sec per machine
        if result == float('inf'):
            return float('inf')
        total += result
    return total
