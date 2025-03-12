#!/usr/bin/env python3

import math

class PurePursuit:
    def __init__(self, look_ahead_dist, wheel_base):
        self.look_ahead_dist = look_ahead_dist
        self.wheel_base = wheel_base

        self.path = []  # global path
        self.current_pos = (0.0, 0.0)
        self.current_yaw = 0.0  # radian
        self.look_ahead_point = None

    def normalize_angle(self, angle):
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle <= math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def get_nearest_index(self, path, x, y):
        # Find the nearest path index to (x, y)
        if not path:
            return None
        dists = [(x - px)**2 + (y - py)**2 for (px, py) in path]
        min_dist = min(dists)
        return dists.index(min_dist)

    def get_look_ahead_point(self, path, x, y, nearest_index, look_ahead_dist):
        # Look ahead from nearest index until distance is >= look_ahead_dist
        if not path:
            return None
        for i in range(nearest_index + 1, len(path)):
            px, py = path[i]
            dist_val = math.sqrt((px - x)**2 + (py - y)**2)
            if dist_val >= look_ahead_dist:
                return (px, py)
        return None

    def compute_steering_angle(self, path, current_pos, current_yaw, look_ahead_dist=None, wheel_base=None):
        # Use the vehicle yaw and the position of the look-ahead point
        if not path:
            self.look_ahead_point = None
            return 0.0
        
        if look_ahead_dist is None:
            look_ahead_dist = self.look_ahead_dist
        if wheel_base is None:
            wheel_base = self.wheel_base

        x, y = current_pos
        nearest_index = self.get_nearest_index(path, x, y)
        if nearest_index is None:
            self.look_ahead_point = None
            return 0.0

        look_ahead_point = self.get_look_ahead_point(path, x, y, nearest_index, look_ahead_dist)
        self.look_ahead_point = look_ahead_point
        if not look_ahead_point:
            return 0.0

        lx, ly = look_ahead_point
        angle_to_target = math.atan2(ly - y, lx - x)
        heading_error = angle_to_target - current_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        steering_angle = math.atan2(
            2.0 * wheel_base * math.sin(heading_error),
            look_ahead_dist
        )

        return steering_angle
