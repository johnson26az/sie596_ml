import heapq

from networkx import neighbors

def astar(grid, start, goal):

    h, w = grid.shape
    def neighbors(ix, iy):
        for dx, dy, in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = ix+dx, iy+dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                yield (nx, ny)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nb in neighbors(*current):
            tentative_g_score = cost + 1
            if nb not in g_score or tentative_g_score < g_score[nb]:
                g_score[nb] = tentative_g_score
                came_from[nb] = current
                f_score = tentative_g_score + heuristic(nb, goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, nb))

    return [start]  # no path found, return start as fallback