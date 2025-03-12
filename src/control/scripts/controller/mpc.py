#!/usr/bin/env python3

import rospy
import math
import numpy as np
import json
import casadi as ca
import heapq

from scipy.interpolate import CubicSpline
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Bool
from utils.msg import localisation

class StateStruct:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

# ------------------ MPC Controller ------------------
class NonlinearMPCController:
    def __init__(self, dt=0.25, horizon=8, wheelbase=0.26):
        self.dt = dt
        self.T = horizon
        self.wb = wheelbase
        self.scenario = 'driving'
        self.set_scenario(self.scenario)
        self.nx = 3  # (x, y, yaw)
        self.nu = 2  # (v, steer)

        # JJB
        self.global_idx = 0
        self.parking_idx = 0
        self.exit_parking_idx = 0

    def set_scenario(self, scenario):
        self.scenario = scenario
        if scenario == "parking" or scenario == "exit_parking":
            rospy.loginfo("[MPC] => parking param set")
            self.T = 12  # 예측 지평선 증가
            self.Qx = 40.0
            self.Qy = 70.0
            self.Qyaw = 2
            self.Rv = 0.005
            self.Rsteer = 0.005
            self.Rdv = 0.001
            self.Rdsteer = 0.001
            self.v_min = -0.25
            self.v_max = 0.2
            self.max_steer = math.radians(25.0)
            self.min_steer = -self.max_steer
        else:
            rospy.loginfo("[MPC] => driving param set")
            self.T = 12
            self.Qx = 700.0
            self.Qy = 700.0
            self.Qyaw = 500.0
            self.Rv = 0.01
            self.Rsteer = 0.01
            self.Rdv = 0.005
            self.Rdsteer = 0.005
            self.v_min = -0.2
            self.v_max = 0.3
            self.max_steer = math.radians(25.0)
            self.min_steer = -self.max_steer

    def solve_mpc(self, x0, xref):
        T = self.T
        dt = self.dt
        wb = self.wb
        n_vars = (self.nx)*(T+1) + (self.nu)*T
        opt_x = ca.SX.sym('opt_x', n_vars)

        def sidx(k): return self.nx*k
        def cidx(k): return self.nx*(T+1) + self.nu*k

        obj = 0.0
        g = []
        lbg = []
        ubg = []

        # 초기 상태 고정
        g += [opt_x[sidx(0)+0] - x0[0], opt_x[sidx(0)+1] - x0[1], opt_x[sidx(0)+2] - x0[2]]
        lbg += [0.0, 0.0, 0.0]
        ubg += [0.0, 0.0, 0.0]

        def f(st, con):
            x, y, yaw = st[0], st[1], st[2]
            v, steer = con[0], con[1]
            dx = v * ca.cos(yaw)
            dy = v * ca.sin(yaw)
            dyaw = (v/wb)*steer
            return ca.vertcat(dx, dy, dyaw)

        for k in range(T):
            st_k = opt_x[sidx(k):sidx(k)+3]
            con_k = opt_x[cidx(k):cidx(k)+2]
            st_next = opt_x[sidx(k+1):sidx(k+1)+3]
            k1 = f(st_k, con_k)
            k2 = f(st_k+(dt/2)*k1, con_k)
            k3 = f(st_k+(dt/2)*k2, con_k)
            k4 = f(st_k+dt*k3, con_k)
            st_rk4 = st_k+(dt/6)*(k1+2*k2+2*k3+k4)
            g += [st_next - st_rk4]
            lbg += [0.0, 0.0, 0.0]
            ubg += [0.0, 0.0, 0.0]

        for k in range(T+1):
            xk = opt_x[sidx(k)+0]
            yk = opt_x[sidx(k)+1]
            yawk = opt_x[sidx(k)+2]
            xr = xref[0, k]
            yr = xref[1, k]
            yr_yaw = xref[2, k]
            obj += self.Qx*(xk - xr)**2 + self.Qy*(yk - yr)**2 + self.Qyaw*(yawk - yr_yaw)**2

        for k in range(T):
            vk = opt_x[cidx(k)+0]
            steer_k = opt_x[cidx(k)+1]
            obj += self.Rv*(vk**2) + self.Rsteer*(steer_k**2)
            if k < T-1:
                v_next = opt_x[cidx(k+1)+0]
                s_next = opt_x[cidx(k+1)+1]
                obj += self.Rdv*((v_next - vk)**2) + self.Rdsteer*((s_next - steer_k)**2)

        lbx = [-1e6]*((T+1)*self.nx) + [self.v_min, self.min_steer]*T
        ubx = [1e6]*((T+1)*self.nx) + [self.v_max, self.max_steer]*T

        nlp = {'f': obj, 'x': opt_x, 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt': {'max_iter': 200, 'acceptable_tol': 1e-6, 'acceptable_obj_change_tol': 1e-6, 'print_level': 0}, 'print_time': 0
        })

        x_init = []
        for k in range(T+1):
            alpha = k/float(T+1)
            xg = x0[0]*(1-alpha) + xref[0, -1]*alpha
            yg = x0[1]*(1-alpha) + xref[1, -1]*alpha
            yawg = x0[2]*(1-alpha) + xref[2, -1]*alpha
            x_init += [xg, yg, yawg]
        x_init += [0.0, 0.0]*T

        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        if sol['f'].full()[0] > 1e6 or 'Solve_Succeeded' not in solver.stats()['return_status']:
            rospy.logwarn(f"[MPC] solver fail: {solver.stats()['return_status']}")
            return None, None

        solx = sol['x'].full().flatten()
        all_controls = solx[self.nx*(T+1):].reshape((T, self.nu)).T
        return (all_controls[0, :], all_controls[1, :]), None
    
    # JJB
    def apply_cubic_spline(self, path_coords):
        if len(path_coords) < 3:
            return path_coords
        path_array = np.array(path_coords)
        distances = np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1))
        t = np.concatenate(([0], np.cumsum(distances)))
        cs_x = CubicSpline(t, path_array[:, 0])
        cs_y = CubicSpline(t, path_array[:, 1])
        t_new = np.linspace(0, t[-1], num=100)
        new_x = cs_x(t_new)
        new_y = cs_y(t_new)
        new_path = list(zip(new_x, new_y))
        return new_path
    
    def build_parking_path(self, start_node, end_node, graph_nodes, graph_edges):
        import heapq
        dist = {n: float('inf') for n in graph_nodes.keys()}
        prev = {n: None for n in graph_nodes.keys()}
        dist[start_node] = 0.0
        pq = [(0.0, start_node)]
        while pq:
            cur_dist, cur_node = heapq.heappop(pq)
            if cur_dist > dist[cur_node]:
                continue
            if cur_node == end_node:
                break
            if cur_node not in graph_edges:
                continue
            neighbors = graph_edges[cur_node]
            for nxt in neighbors:
                cost = math.hypot(graph_nodes[nxt][0] - graph_nodes[cur_node][0],
                                  graph_nodes[nxt][1] - graph_nodes[cur_node][1])
                alt = dist[cur_node] + cost
                if alt < dist[nxt]:
                    dist[nxt] = alt
                    prev[nxt] = cur_node
                    heapq.heappush(pq, (alt, nxt))
        path_nodes = []
        n = end_node
        while n is not None:
            path_nodes.append(n)
            n = prev[n]
        path_nodes.reverse()
        raw_path = [graph_nodes[n] for n in path_nodes]
        return self.apply_cubic_spline(raw_path)
    
    def get_nearest_idx(self, x, y, path, start_i):
        if start_i >= len(path):
            return len(path) - 1
        return min(range(start_i, len(path)), key=lambda i: (x - path[i][0])**2 + (y - path[i][1])**2)
    
    def build_xref(self, path_xy, near_i, st, is_parking=False):
        T = self.T
        xref = np.zeros((3, T+1))
        n = len(path_xy)
        for i in range(T+1):
            idx = min(near_i + i, n-1)
            xref[0, i] = path_xy[idx][0]
            xref[1, i] = path_xy[idx][1]
            if i == 0:
                xref[2, i] = st.yaw
            else:
                if is_parking and idx == n-1:
                    xref[2, i] = 0.0
                else:
                    dx = path_xy[idx][0] - path_xy[idx-1][0]
                    dy = path_xy[idx][1] - path_xy[idx-1][1]
                    prev_yaw = xref[2, i-1]
                    new_yaw = math.atan2(dy, dx)
                    xref[2, i] = prev_yaw + math.atan2(math.sin(new_yaw - prev_yaw),
                                                       math.cos(new_yaw - prev_yaw))
        return xref
        
    def compute_control_command(self, path, current_pos, current_yaw, desired_speed, scenario="driving"):
        self.x, self.y = current_pos
        self.yaw = current_yaw
        self.set_scenario(scenario)
        self.v_max = desired_speed

        st = StateStruct(self.x, self.y, self.yaw)

        if scenario == "driving":
            # JJB
            self.global_path = path

            # mechsoon
            near_i = self.get_nearest_idx(st.x, st.y, self.global_path, self.global_idx)
            self.global_idx = near_i
            xref = self.build_xref(self.global_path, near_i, st, is_parking=False)
            x0 = np.array([st.x, st.y, st.yaw])
            (v_traj, steer_traj), _ = self.solve_mpc(x0, xref)
            v_cmd = v_traj[0] if v_traj is not None else 0.0
            s_cmd = steer_traj[0] if steer_traj is not None else 0.0

        elif scenario == "lane_change":
            # JJB
            self.lane_change_path = path

            # mechsoon
            near_i = self.get_nearest_idx(st.x, st.y, self.lane_change_path, self.lane_change_idx)
            self.lane_change_idx = near_i
            xref = self.build_xref(self.lane_change_path, near_i, st, is_parking=False)
            x0 = np.array([st.x, st.y, st.yaw])
            (v_traj, steer_traj), _ = self.mpc.solve_mpc(x0, xref)
            v_cmd = v_traj[0] if v_traj is not None else 0.0
            s_cmd = steer_traj[0] if steer_traj is not None else 0.0

        elif scenario == "parking":
            # JJB
            self.parking_path = path

            # mechsoon
            near_i = self.get_nearest_idx(st.x, st.y, self.parking_path, self.parking_idx)
            self.parking_idx = near_i
            xref = self.build_xref(self.parking_path, near_i, st, is_parking=True)
            x0 = np.array([st.x, st.y, st.yaw])
            (v_traj, steer_traj), _ = self.solve_mpc(x0, xref)
            v_cmd = v_traj[0] if v_traj is not None else 0.0
            s_cmd = steer_traj[0] if steer_traj is not None else 0.0

        elif scenario == "exit_parking":
            # JJB
            self.exit_parking_path = path

            # mechsoon
            near_i = self.get_nearest_idx(st.x, st.y, self.exit_parking_path, self.exit_parking_idx)
            self.exit_parking_idx = near_i
            xref = self.build_xref(self.exit_parking_path, near_i, st, is_parking=True)
            x0 = np.array([st.x, st.y, st.yaw])
            (v_traj, steer_traj), _ = self.solve_mpc(x0, xref)
            v_cmd = v_traj[0] if v_traj is not None else 0.0
            s_cmd = steer_traj[0] if steer_traj is not None else 0.0

        return v_cmd, s_cmd
