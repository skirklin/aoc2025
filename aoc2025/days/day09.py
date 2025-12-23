"""Day 9: Movie Theater"""


def parse(data: str):
    """Parse input into list of red tile coordinates."""
    tiles = []
    for line in data.strip().split('\n'):
        x, y = map(int, line.split(','))
        tiles.append((x, y))
    return tiles


def rectangle_area(p1, p2):
    """Calculate rectangle area with p1 and p2 as opposite corners (inclusive)."""
    width = abs(p2[0] - p1[0]) + 1
    height = abs(p2[1] - p1[1]) + 1
    return width * height


def part1(data: str) -> int:
    """Find largest rectangle with red tiles at opposite corners."""
    tiles = parse(data)
    max_area = 0

    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            area = rectangle_area(tiles[i], tiles[j])
            max_area = max(max_area, area)

    return max_area


def build_polygon_info(tiles):
    """Build polygon boundary and compute valid x-ranges for each y."""
    n = len(tiles)
    edges = []
    for i in range(n):
        p1 = tiles[i]
        p2 = tiles[(i + 1) % n]
        edges.append((p1, p2))

    # Collect all unique y-coordinates (including midpoints for interior check)
    ys = sorted(set(t[1] for t in tiles))

    # For each y, compute the valid x-ranges using scan-line
    def get_x_ranges(y):
        """Get list of (x_start, x_end) ranges that are inside/on polygon at height y."""
        # Find all x-coordinates where vertical edges cross this y
        crossings = []
        for (x1, y1), (x2, y2) in edges:
            if x1 == x2:  # Vertical edge
                if min(y1, y2) <= y <= max(y1, y2):
                    crossings.append(x1)

        crossings.sort()

        # Also track horizontal edges at this y
        h_edges = []
        for (x1, y1), (x2, y2) in edges:
            if y1 == y2 == y:  # Horizontal edge at this y
                h_edges.append((min(x1, x2), max(x1, x2)))

        # Combine crossings with horizontal edges
        # The interior is between pairs of crossings
        ranges = []
        for i in range(0, len(crossings) - 1, 2):
            ranges.append((crossings[i], crossings[i + 1]))

        return ranges

    return edges, ys, get_x_ranges


def rect_valid(x1, y1, x2, y2, edges):
    """Check if rectangle is entirely inside/on the polygon."""
    # Check all four corners
    for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
        if not point_inside_or_on(x, y, edges):
            return False

    # Check if rectangle crosses boundary going outside
    # For orthogonal polygon, check if all edges of rectangle are inside
    # Sample points along rectangle boundary
    for x in [x1, x2]:
        for y in range(y1, y2 + 1):
            if not point_inside_or_on(x, y, edges):
                return False
    for y in [y1, y2]:
        for x in range(x1, x2 + 1):
            if not point_inside_or_on(x, y, edges):
                return False

    return True


def point_inside_or_on(x, y, edges):
    """Check if point is inside or on the polygon boundary."""
    # Check if on boundary
    for (x1, y1), (x2, y2) in edges:
        if x1 == x2 == x and min(y1, y2) <= y <= max(y1, y2):
            return True
        if y1 == y2 == y and min(x1, x2) <= x <= max(x1, x2):
            return True

    # Ray casting for interior
    crossings = 0
    for (x1, y1), (x2, y2) in edges:
        if y1 == y2:  # Horizontal edge
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if y < y1 or y >= y2:
            continue
        t = (y - y1) / (y2 - y1)
        x_int = x1 + t * (x2 - x1)
        if x < x_int:
            crossings += 1
    return crossings % 2 == 1


def part2(data: str) -> int:
    """Find largest rectangle using only red/green tiles."""
    tiles = parse(data)
    n = len(tiles)

    # Build edges
    edges = []
    for i in range(n):
        p1 = tiles[i]
        p2 = tiles[(i + 1) % n]
        edges.append((p1, p2))

    # Collect unique x and y coordinates
    xs = sorted(set(t[0] for t in tiles))
    ys = sorted(set(t[1] for t in tiles))

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    # Build vertical edge events
    v_edges = []  # (y_min, y_max, x)
    for (x1, y1), (x2, y2) in edges:
        if x1 == x2:
            v_edges.append((min(y1, y2), max(y1, y2), x1))

    def is_inside(x, y):
        """Check if point is strictly inside polygon using ray casting."""
        crossings = 0
        for y_min, y_max, ex in v_edges:
            if y_min <= y < y_max and ex < x:
                crossings += 1
        return crossings % 2 == 1

    # Build compressed grid: cell[i][j] represents region
    # xs[i] <= x < xs[i+1] and ys[j] <= y < ys[j+1]
    # A cell is "valid" if its interior is inside the polygon
    num_x = len(xs)
    num_y = len(ys)

    # For each cell, check if the center point is inside
    cell_valid = [[False] * num_y for _ in range(num_x)]
    for i in range(num_x - 1):
        for j in range(num_y - 1):
            mid_x = (xs[i] + xs[i + 1]) / 2
            mid_y = (ys[j] + ys[j + 1]) / 2
            cell_valid[i][j] = is_inside(mid_x, mid_y)

    # Build prefix sum for fast rectangle queries
    prefix = [[0] * (num_y + 1) for _ in range(num_x + 1)]
    for i in range(num_x):
        for j in range(num_y):
            val = 1 if cell_valid[i][j] else 0
            prefix[i + 1][j + 1] = val + prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j]

    def count_valid_cells(xi1, yi1, xi2, yi2):
        """Count valid cells in range [xi1, xi2) x [yi1, yi2)."""
        return (prefix[xi2][yi2] - prefix[xi1][yi2] -
                prefix[xi2][yi1] + prefix[xi1][yi1])

    def rect_fully_inside(rx1, ry1, rx2, ry2):
        """Check if rectangle is fully inside polygon."""
        # Get compressed indices
        xi1 = x_to_idx.get(rx1)
        yi1 = y_to_idx.get(ry1)
        xi2 = x_to_idx.get(rx2)
        yi2 = y_to_idx.get(ry2)

        if None in (xi1, yi1, xi2, yi2):
            return False

        # Count cells and expected cells
        expected = (xi2 - xi1) * (yi2 - yi1)
        actual = count_valid_cells(xi1, yi1, xi2, yi2)
        return actual == expected

    max_area = 0

    # Check all pairs of red tiles
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            p1, p2 = tiles[i], tiles[j]
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area <= max_area:
                continue

            if rect_fully_inside(x1, y1, x2, y2):
                max_area = area

    return max_area
