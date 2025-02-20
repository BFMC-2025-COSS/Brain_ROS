#!/usr/bin/env python3

def get_nearest_index(path, x, y):
    # Find the nearest path index to (x, y)
    if not path:
        return None
    dists = [(x - px)**2 + (y - py)**2 for (px, py) in path]
    min_dist = min(dists)
    return dists.index(min_dist)

def get_index(path, x, y):
    for i, (path_x, path_y) in enumerate(path):
        if x == path_x and y == path_y:
            return i
    return None 
