"""Day 8: Playground"""
import math
from collections import Counter


def parse(data: str):
    """Parse input into list of 3D coordinates."""
    points = []
    for line in data.strip().split('\n'):
        x, y, z = map(int, line.split(','))
        points.append((x, y, z))
    return points


def distance_sq(p1, p2):
    """Calculate squared Euclidean distance between two points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in same set
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_sizes(self):
        sizes = []
        for i in range(len(self.parent)):
            if self.parent[i] == i:
                sizes.append(self.size[i])
        return sizes


def solve(points, num_connections):
    """Connect num_connections closest pairs and return product of 3 largest circuit sizes."""
    n = len(points)

    # Calculate all pairwise distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_sq(points[i], points[j])
            edges.append((d, i, j))

    # Sort by distance
    edges.sort()

    # Union-find to make connections
    uf = UnionFind(n)
    connections = 0
    for d, i, j in edges:
        if connections >= num_connections:
            break
        uf.union(i, j)  # Connect even if already in same circuit
        connections += 1

    # Get circuit sizes
    sizes = uf.get_sizes()
    sizes.sort(reverse=True)

    return sizes[0] * sizes[1] * sizes[2]


def part1(data: str) -> int:
    """Connect 1000 closest pairs, return product of 3 largest circuit sizes."""
    points = parse(data)
    return solve(points, 1000)


def part2(data: str) -> int:
    """Find last connection to form single circuit, return product of X coords."""
    points = parse(data)
    n = len(points)

    # Calculate all pairwise distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_sq(points[i], points[j])
            edges.append((d, i, j))

    # Sort by distance
    edges.sort()

    # Union-find to connect until all in one circuit
    uf = UnionFind(n)
    last_i, last_j = None, None

    for d, i, j in edges:
        if uf.union(i, j):  # Actually merged two circuits
            last_i, last_j = i, j
            # Check if all connected
            if uf.size[uf.find(i)] == n:
                break

    return points[last_i][0] * points[last_j][0]
