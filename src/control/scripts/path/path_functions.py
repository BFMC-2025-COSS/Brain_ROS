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

def get_indices_within_distance(path, x, y, dist=0.1):
    indices = []
    dist_sq = dist * dist

    for i, (px, py) in enumerate(path):
        dx = px - x
        dy = py - y
        if (dx*dx + dy*dy) <= dist_sq:
            indices.append(i)
    
    return indices
