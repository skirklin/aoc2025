"""Day 12: Christmas Tree Farm"""


def parse(data: str):
    """Parse input into shapes and regions."""
    parts = data.strip().split('\n\n')

    # Parse shapes
    shapes = {}
    for part in parts[:-1]:  # All but last part are shapes
        lines = part.strip().split('\n')
        idx = int(lines[0].split(':')[0])
        shape = []
        for r, line in enumerate(lines[1:]):
            for c, ch in enumerate(line):
                if ch == '#':
                    shape.append((r, c))
        shapes[idx] = shape

    # Parse regions
    regions = []
    for line in parts[-1].strip().split('\n'):
        parts2 = line.split(': ')
        dims = parts2[0].split('x')
        width, height = int(dims[0]), int(dims[1])
        counts = list(map(int, parts2[1].split()))
        regions.append((width, height, counts))

    return shapes, regions


def get_rotations(shape):
    """Get all 8 rotations/flips of a shape."""
    rotations = set()

    def normalize(s):
        min_r = min(p[0] for p in s)
        min_c = min(p[1] for p in s)
        return tuple(sorted((r - min_r, c - min_c) for r, c in s))

    def rotate90(s):
        return [(c, -r) for r, c in s]

    def flip(s):
        return [(-r, c) for r, c in s]

    current = shape
    for _ in range(4):
        rotations.add(normalize(current))
        rotations.add(normalize(flip(current)))
        current = rotate90(current)

    return list(rotations)


def can_place(grid, shape, r, c, width, height):
    """Check if shape can be placed at position (r, c)."""
    for dr, dc in shape:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= height or nc < 0 or nc >= width:
            return False
        if grid[nr][nc]:
            return False
    return True


def place(grid, shape, r, c):
    """Place shape at position (r, c)."""
    for dr, dc in shape:
        grid[r + dr][c + dc] = True


def unplace(grid, shape, r, c):
    """Remove shape from position (r, c)."""
    for dr, dc in shape:
        grid[r + dr][c + dc] = False


def solve_region(width, height, shapes, counts, timeout_sec=60):
    """Try to fit all presents into the region."""
    import time
    start_time = time.time()

    # Build list of shapes to place (with all rotations)
    pieces = []
    for idx, count in enumerate(counts):
        if idx in shapes:
            rotations = get_rotations(shapes[idx])
            for _ in range(count):
                pieces.append((idx, rotations))

    if not pieces:
        return True  # No pieces to place

    total_cells = sum(len(shapes[idx]) * counts[idx]
                      for idx, count in enumerate(counts) if idx in shapes and count > 0)
    if total_cells > width * height:
        return False  # Quick check: not enough space

    # Initialize empty grid
    grid = [[False] * width for _ in range(height)]

    # Generate all valid placements for each piece
    def get_all_placements(rotations):
        """Get all valid placements for a piece."""
        placements = []
        for shape in rotations:
            for r in range(height):
                for c in range(width):
                    if can_place(grid, shape, r, c, width, height):
                        # Check it actually works by temporarily placing
                        pass
                    placements.append((shape, r, c))
        return placements

    def backtrack(piece_idx):
        if time.time() - start_time > timeout_sec:
            return None  # Timeout

        if piece_idx == len(pieces):
            return True  # All pieces placed

        idx, rotations = pieces[piece_idx]

        # Try all positions for all rotations
        for shape in rotations:
            for r in range(height):
                for c in range(width):
                    if can_place(grid, shape, r, c, width, height):
                        place(grid, shape, r, c)
                        result = backtrack(piece_idx + 1)
                        if result is True:
                            return True
                        if result is None:
                            return None  # Propagate timeout
                        unplace(grid, shape, r, c)

        return False

    result = backtrack(0)
    return result is True  # Timeout counts as failure


def part1(data: str) -> int:
    """Count regions that can fit all presents."""
    shapes, regions = parse(data)
    count = 0
    for width, height, counts in regions:
        if solve_region(width, height, shapes, counts):
            count += 1
    return count


def part2(data: str) -> int:
    """Solve part 2."""
    # Will implement after seeing part 2
    pass
