"""Day 11: Reactor"""
from functools import lru_cache


def parse(data: str):
    """Parse input into adjacency list."""
    graph = {}
    for line in data.strip().split('\n'):
        parts = line.split(': ')
        node = parts[0]
        outputs = parts[1].split() if len(parts) > 1 else []
        graph[node] = outputs
    return graph


def count_paths(graph, start, end):
    """Count all paths from start to end using memoization."""
    @lru_cache(maxsize=None)
    def dfs(node):
        if node == end:
            return 1
        if node not in graph:
            return 0
        return sum(dfs(child) for child in graph[node])

    return dfs(start)


def part1(data: str) -> int:
    """Count paths from 'you' to 'out'."""
    graph = parse(data)
    return count_paths(graph, 'you', 'out')


def part2(data: str) -> int:
    """Count paths from svr to out that visit both dac and fft."""
    graph = parse(data)

    # For Part 2, we count paths from svr to out that pass through both dac and fft
    # A path can go: svr -> dac -> fft -> out OR svr -> fft -> dac -> out

    # Count paths between any two nodes
    def count(start, end):
        @lru_cache(maxsize=None)
        def dfs(node):
            if node == end:
                return 1
            if node not in graph:
                return 0
            return sum(dfs(child) for child in graph[node])
        return dfs(start)

    # Clear cache for each count computation
    count.__wrapped__ = True

    # Path through dac first, then fft
    paths_dac_fft = count('svr', 'dac') * count('dac', 'fft') * count('fft', 'out')

    # Path through fft first, then dac
    paths_fft_dac = count('svr', 'fft') * count('fft', 'dac') * count('dac', 'out')

    return paths_dac_fft + paths_fft_dac
