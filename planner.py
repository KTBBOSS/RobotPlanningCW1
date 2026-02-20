"""
planner.py — Path-finding algorithms for Robot Planning Coursework
===================================================================
Contains all planning modules:
  - GridMapper       : 2D occupancy grid + C-space computation (Phase 1)
  - AStarPlanner     : A* / Weighted A* on 8-connected grid   (Phase 2)
  - RRTConnectPlanner: Bidirectional RRT in continuous space   (Phase 2)
  - PathSmoother     : Shortcut + cubic-spline smoothing       (Phase 3)
  - SpaceTimeAStar   : A* in (x,y,t) for dynamic obstacles    (Phase 3)
  - feasibility_check: Verify that a path exists
"""

import time
import heapq
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.interpolate import CubicSpline
from matplotlib.path import Path


# ====================================================================== #
#  Utility
# ====================================================================== #

def euclidean_path_length(path):
    """Total Euclidean length of a waypoint list [(x,y), ...]."""
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        length += np.sqrt(dx * dx + dy * dy)
    return length


# ====================================================================== #
#  GridMapper — workspace + C-space grid representations
# ====================================================================== #

class GridMapper:
    """Manages a 2D occupancy grid for both workspace and C-space."""

    def __init__(self, setup, resolution=0.1):
        self.res = resolution
        self.bounds = setup['map_bounds']                          # [x_min, x_max, y_min, y_max]
        self.width  = int((self.bounds[1] - self.bounds[0]) / self.res)  # cells in x
        self.height = int((self.bounds[3] - self.bounds[2]) / self.res)  # cells in y
        self.grid       = np.zeros((self.width, self.height), dtype=np.float64)
        self.cspace_grid = None

    # Coordinate conversions 
    def world_to_grid(self, x, y):
        """Continuous world coordinates -> discrete grid indices."""
        gx = int(round((x - self.bounds[0]) / self.res))
        gy = int(round((y - self.bounds[2]) / self.res))
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Discrete grid indices -> continuous world coordinates (cell centre)."""
        x = gx * self.res + self.bounds[0] + self.res / 2
        y = gy * self.res + self.bounds[2] + self.res / 2
        return x, y

    # Workspace grid 
    def fill_obstacles(self, obstacles):
        """Fill the workspace grid with axis-aligned rectangular obstacles."""
        for obs in obstacles:
            x_min = obs['pos'][0] - obs['extents'][0]
            x_max = obs['pos'][0] + obs['extents'][0]
            y_min = obs['pos'][1] - obs['extents'][1]
            y_max = obs['pos'][1] + obs['extents'][1]
            gx1, gy1 = self.world_to_grid(x_min, y_min)
            gx2, gy2 = self.world_to_grid(x_max, y_max)
            self.grid[max(0, gx1):min(self.width,  gx2),
                      max(0, gy1):min(self.height, gy2)] = 1

    def overlay_robot(self, x, y, robot_type, robot_geo):
        """Return a copy of the grid with the robot footprint drawn (value 2)."""
        temp_grid = self.grid.copy()
        if robot_type == "RECT":
            hw, hl = robot_geo[0] / 2, robot_geo[1] / 2
            gx1, gy1 = self.world_to_grid(x - hw, y - hl)
            gx2, gy2 = self.world_to_grid(x + hw, y + hl)
            temp_grid[max(0, gx1):min(self.width, gx2),
                      max(0, gy1):min(self.height, gy2)] = 2
        else:
            poly_grid_pts = []
            for pt in robot_geo:
                gx, gy = self.world_to_grid(x + pt[0], y + pt[1])
                poly_grid_pts.append((gx, gy))
            nodes = np.array(poly_grid_pts)
            min_x, min_y = nodes.min(axis=0)
            max_x, max_y = nodes.max(axis=0)
            min_x = max(0, min_x);  min_y = max(0, min_y)
            max_x = min(self.width - 1, max_x)
            max_y = min(self.height - 1, max_y)
            path = Path(poly_grid_pts)
            for ix in range(int(min_x), int(max_x) + 1):
                for iy in range(int(min_y), int(max_y) + 1):
                    if path.contains_point((ix, iy)):
                        temp_grid[ix, iy] = 2
        return temp_grid

    # Phase 1: C-Space Computation 

    def compute_cspace(self, robot_type, robot_geo):
        """
        Task 1.1 — Compute C-space obstacles via Minkowski difference.

        The C-space obstacle set is defined as:

            C_obs = O  ⊖  R  =  O  ⊕  (−R)

        i.e. the Minkowski sum of each workspace obstacle O with the
        *reflected* robot footprint (−R).  This is equivalent to
        morphological dilation of the binary obstacle image using a
        structuring element shaped like (−R).

        For symmetric robots (e.g. axis-aligned rectangles centred at the
        origin) the reflection is a no-op ((−R) = R).  For asymmetric
        polygonal robots the vertices are explicitly negated before
        rasterising the kernel.
        """
        kernel = self._build_robot_kernel(robot_type, robot_geo)
        obstacle_binary = (self.grid > 0).astype(bool)
        cspace_binary   = binary_dilation(obstacle_binary, structure=kernel)

        pad_x, pad_y = self._boundary_padding(robot_type, robot_geo)
        if pad_x > 0:
            cspace_binary[:pad_x,  :] = True
            cspace_binary[-pad_x:, :] = True
        if pad_y > 0:
            cspace_binary[:, :pad_y]  = True
            cspace_binary[:, -pad_y:] = True

        self.cspace_grid = cspace_binary.astype(np.int8)
        return self.cspace_grid

    def _build_robot_kernel(self, robot_type, robot_geo):
        """
        Build the dilation kernel = rasterised (−R).

        For RECT: symmetric about origin, so (−R) = R → simple box kernel.
        For POLY: negate every vertex, then rasterise into a boolean grid.
        """
        if robot_type == "RECT":
            # Rectangle centred at origin is symmetric → (−R) = R
            hw_cells = int(np.ceil((robot_geo[0] / 2.0) / self.res))
            hh_cells = int(np.ceil((robot_geo[1] / 2.0) / self.res))
            kw = 2 * hw_cells + 1
            kh = 2 * hh_cells + 1
            kernel = np.ones((kw, kh), dtype=bool)
            print(f"  [Kernel] RECT  {kw}x{kh} cells  "
                  f"(robot {robot_geo[0]:.2f}x{robot_geo[1]:.2f} m)")
            return kernel
        else:
            # Polygonal robot — reflect vertices: (−R) = {−r : r ∈ R}
            reflected = [(-pt[0], -pt[1]) for pt in robot_geo]
            xs = [v[0] for v in reflected]
            ys = [v[1] for v in reflected]
            half_kw = int(np.ceil(max(abs(min(xs)), abs(max(xs))) / self.res))
            half_kh = int(np.ceil(max(abs(min(ys)), abs(max(ys))) / self.res))
            kw = 2 * half_kw + 1
            kh = 2 * half_kh + 1
            kernel = np.zeros((kw, kh), dtype=bool)
            poly_r = [int(round(v[0] / self.res)) + half_kw for v in reflected]
            poly_c = [int(round(v[1] / self.res)) + half_kh for v in reflected]
            from skimage.draw import polygon as draw_polygon
            rr, cc = draw_polygon(poly_r, poly_c, shape=(kw, kh))
            kernel[rr, cc] = True
            kernel[half_kw, half_kh] = True
            print(f"  [Kernel] POLY  {kw}x{kh} cells  "
                  f"(vertices {[list(v[:2]) for v in robot_geo]})")
            return kernel

    def _boundary_padding(self, robot_type, robot_geo):
        if robot_type == "RECT":
            pad_x = int(np.ceil((robot_geo[0] / 2.0) / self.res))
            pad_y = int(np.ceil((robot_geo[1] / 2.0) / self.res))
        else:
            pad_x = int(np.ceil(max(abs(v[0]) for v in robot_geo) / self.res))
            pad_y = int(np.ceil(max(abs(v[1]) for v in robot_geo) / self.res))
        return pad_x, pad_y

    # Task 1.2: Collision checking 

    def check_collision(self, x, y):
        """Return True if robot centred at (x, y) is in C-space collision."""
        if self.cspace_grid is None:
            raise RuntimeError("C-space not computed — call compute_cspace() first.")
        gx, gy = self.world_to_grid(x, y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True
        return bool(self.cspace_grid[gx, gy])


# ====================================================================== #
#  Phase 2: A* / Weighted A*
# ====================================================================== #

class AStarPlanner:
    """
    A* and Weighted A* on the C-space grid.

    - 8-connected movement (cardinal cost 1, diagonal cost sqrt(2)).
    - Heuristic: Euclidean distance (admissible — never overestimates).
    - Corner-cutting prevention on diagonal moves.
    """

    SQRT2 = float(np.sqrt(2))
    MOVES = [
        ( 1,  0, 1.0),    (-1,  0, 1.0),
        ( 0,  1, 1.0),    ( 0, -1, 1.0),
        ( 1,  1, SQRT2),  ( 1, -1, SQRT2),
        (-1,  1, SQRT2),  (-1, -1, SQRT2),
    ]

    def __init__(self, mapper):
        self.mapper = mapper
        self.cspace = mapper.cspace_grid
        self.W = mapper.width
        self.H = mapper.height

    @staticmethod
    def heuristic(a, b):
        """Euclidean distance — admissible for 8-connected grids."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx * dx + dy * dy)

    def _blocked(self, cell):
        gx, gy = cell
        if gx < 0 or gx >= self.W or gy < 0 or gy >= self.H:
            return True
        return bool(self.cspace[gx, gy])

    def plan(self, start_world, goal_world, weight=1.0):
        """
        Run (Weighted) A*.

        Returns (path_world, stats).
        path_world: list[(x, y)] in world coords, or None.
        stats: dict with success, time_ms, path_length, nodes_expanded, weight.
        """
        start = self.mapper.world_to_grid(*start_world)
        goal  = self.mapper.world_to_grid(*goal_world)
        if self._blocked(start) or self._blocked(goal):
            return None, {'success': False, 'reason': 'start/goal in collision'}

        t0 = time.perf_counter()
        open_heap = []
        counter   = 0
        g_cost    = {start: 0.0}
        parent    = {}
        closed    = set()
        nodes_exp = 0

        heapq.heappush(open_heap,
                        (weight * self.heuristic(start, goal), counter, start))
        counter += 1

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            nodes_exp += 1

            if current == goal:
                path_grid = []
                node = current
                while node in parent:
                    path_grid.append(node)
                    node = parent[node]
                path_grid.append(start)
                path_grid.reverse()
                path_world = [self.mapper.grid_to_world(*c) for c in path_grid]
                elapsed = (time.perf_counter() - t0) * 1000
                return path_world, {
                    'success': True, 'time_ms': elapsed,
                    'path_length': euclidean_path_length(path_world),
                    'nodes_expanded': nodes_exp, 'weight': weight,
                    'expanded_cells': closed,
                }

            for dx, dy, step_cost in self.MOVES:
                nb = (current[0] + dx, current[1] + dy)
                if nb in closed or self._blocked(nb):
                    continue
                if dx != 0 and dy != 0:
                    if self._blocked((current[0] + dx, current[1])) or \
                       self._blocked((current[0], current[1] + dy)):
                        continue
                tent_g = g_cost[current] + step_cost
                if tent_g < g_cost.get(nb, float('inf')):
                    g_cost[nb] = tent_g
                    parent[nb] = current
                    f = tent_g + weight * self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f, counter, nb))
                    counter += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return None, {'success': False, 'time_ms': elapsed,
                      'nodes_expanded': nodes_exp, 'reason': 'no path found'}


# ====================================================================== #
#  Phase 2: RRT-Connect
# ====================================================================== #

class RRTConnectPlanner:
    """
    Bidirectional RRT-Connect in continuous 2-D C-space.

    - Two trees grow from start and goal.
    - Line-of-sight collision checker samples at sub-resolution intervals.
    - Configurable step_size parameter.
    """

    def __init__(self, mapper, step_size=0.5, max_iter=10000):
        self.mapper    = mapper
        self.step_size = step_size
        self.max_iter  = max_iter
        self.bounds = mapper.bounds

    def plan(self, start, goal):
        if self.mapper.check_collision(*start) or \
           self.mapper.check_collision(*goal):
            return None, {'success': False, 'reason': 'start/goal in collision'}

        t0 = time.perf_counter()
        tree_s = [[start[0], start[1], -1]]
        tree_g = [[goal[0],  goal[1],  -1]]
        verts_sampled = 0

        for i in range(self.max_iter):
            q_rand = self._sample()
            verts_sampled += 1
            if i % 2 == 0:
                ok, q_new = self._extend(tree_s, q_rand)
                if ok and self._connect(tree_g, q_new):
                    path = self._build_path(tree_s, tree_g)
                    return path, self._stats(t0, path, verts_sampled, tree_s, tree_g)
            else:
                ok, q_new = self._extend(tree_g, q_rand)
                if ok and self._connect(tree_s, q_new):
                    path = self._build_path(tree_s, tree_g)
                    return path, self._stats(t0, path, verts_sampled, tree_s, tree_g)

        elapsed = (time.perf_counter() - t0) * 1000
        return None, {'success': False, 'time_ms': elapsed,
                      'vertices_sampled': verts_sampled,
                      'tree_size': len(tree_s) + len(tree_g),
                      'reason': 'max iterations'}

    def _sample(self):
        return (np.random.uniform(self.bounds[0], self.bounds[1]),
                np.random.uniform(self.bounds[2], self.bounds[3]))

    def _nearest_idx(self, tree, point):
        pts = np.array([[n[0], n[1]] for n in tree])
        diff = pts - np.array(point)
        return int(np.argmin(np.sum(diff * diff, axis=1)))

    def _steer(self, from_pt, to_pt):
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        d  = np.hypot(dx, dy)
        if d <= self.step_size:
            return to_pt
        r = self.step_size / d
        return (from_pt[0] + r * dx, from_pt[1] + r * dy)

    def _collision_free_line(self, p1, p2):
        """Line-of-sight check at half-resolution intervals."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d  = np.hypot(dx, dy)
        if d < 1e-9:
            return not self.mapper.check_collision(p1[0], p1[1])
        n_checks = max(2, int(np.ceil(d / (self.mapper.res * 0.5))))
        for k in range(n_checks + 1):
            t = k / n_checks
            if self.mapper.check_collision(p1[0] + t * dx, p1[1] + t * dy):
                return False
        return True

    def _extend(self, tree, target):
        ni   = self._nearest_idx(tree, target)
        near = (tree[ni][0], tree[ni][1])
        new  = self._steer(near, target)
        if self._collision_free_line(near, new):
            tree.append([new[0], new[1], ni])
            return True, new
        return False, None

    def _connect(self, tree, target, max_steps=300):
        for _ in range(max_steps):
            ok, q = self._extend(tree, target)
            if not ok:
                return False
            if np.hypot(q[0] - target[0], q[1] - target[1]) < 1e-3:
                return True
        return False

    def _build_path(self, tree_s, tree_g):
        path_s = []
        idx = len(tree_s) - 1
        while idx != -1:
            path_s.append((tree_s[idx][0], tree_s[idx][1]))
            idx = tree_s[idx][2]
        path_s.reverse()
        path_g = []
        idx = len(tree_g) - 1
        while idx != -1:
            path_g.append((tree_g[idx][0], tree_g[idx][1]))
            idx = tree_g[idx][2]
        return path_s + path_g[1:]

    def _stats(self, t0, path, verts, tree_s, tree_g):
        return {'success': True,
                'time_ms': (time.perf_counter() - t0) * 1000,
                'path_length': euclidean_path_length(path),
                'vertices_sampled': verts,
                'tree_size': len(tree_s) + len(tree_g)}


# ====================================================================== #
#  Phase 3: Path Smoothing (shortcut + cubic spline)
# ====================================================================== #

class PathSmoother:
    """Greedy shortcutting followed by cubic-spline fitting (C2 continuous)."""

    def __init__(self, mapper):
        self.mapper = mapper

    def smooth(self, path, num_points=300):
        if len(path) < 3:
            return path
        sc = self._shortcut(path)
        sp = self._fit_spline(sc, num_points)
        if self._collision_free(sp):
            return sp
        for skip in [10, 5, 3, 2, 1]:
            ctrl = path[::skip]
            if ctrl[-1] != path[-1]:
                ctrl.append(path[-1])
            sp = self._fit_spline(ctrl, num_points)
            if self._collision_free(sp):
                return sp
        return path

    def _shortcut(self, path):
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_free(path[i], path[j]):
                    break
                j -= 1
            result.append(path[j])
            i = j
        return result

    def _fit_spline(self, pts_list, num_points):
        pts = np.array(pts_list)
        diffs = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        mask = np.concatenate([[True], np.diff(cum) > 1e-9])
        cum = cum[mask];  pts = pts[mask]
        if len(pts) < 3:
            s = np.linspace(0, 1, num_points)
            xs = np.interp(s, [0, 1], [pts[0, 0], pts[-1, 0]])
            ys = np.interp(s, [0, 1], [pts[0, 1], pts[-1, 1]])
            return list(zip(xs.tolist(), ys.tolist()))
        cs_x = CubicSpline(cum, pts[:, 0], bc_type='natural')
        cs_y = CubicSpline(cum, pts[:, 1], bc_type='natural')
        s_new = np.linspace(0, cum[-1], num_points)
        return list(zip(cs_x(s_new).tolist(), cs_y(s_new).tolist()))

    def _line_free(self, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        d = np.hypot(dx, dy)
        if d < 1e-9:
            return not self.mapper.check_collision(p1[0], p1[1])
        n = max(2, int(np.ceil(d / (self.mapper.res * 0.5))))
        for k in range(n + 1):
            t = k / n
            if self.mapper.check_collision(p1[0] + t * dx, p1[1] + t * dy):
                return False
        return True

    def _collision_free(self, path):
        return all(not self.mapper.check_collision(x, y) for x, y in path)


# ====================================================================== #
#  Phase 3: Space-Time A* (dynamic obstacle avoidance)
# ====================================================================== #

class SpaceTimeAStar:
    """
    A* in (x, y, t) state space for dynamic obstacle avoidance.

    The moving obstacle's patrol trajectory is fully known, so its position
    is predicted analytically at every future time step.

    Actions: 8-connected moves + wait-in-place (all cost 1 time step).
    Heuristic: Chebyshev distance (admissible for unit-time steps).
    """

    MOVES = [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1),
        ( 1,  1), ( 1, -1), (-1,  1), (-1, -1),
        ( 0,  0),           # wait
    ]

    def __init__(self, mapper, dyn_obs_info, robot_type, robot_geo,
                 robot_speed=1.0, sim_freq=240):
        self.mapper     = mapper
        self.dyn_info   = dyn_obs_info
        self.robot_speed = robot_speed
        self.sim_freq   = sim_freq
        self.dt = mapper.res / robot_speed

        dyn_r = dyn_obs_info['radius']
        if robot_type == "RECT":
            robot_r = np.hypot(robot_geo[0] / 2, robot_geo[1] / 2)
        else:
            robot_r = max(np.hypot(v[0], v[1]) for v in robot_geo)
        self.inflated_r_sq = (dyn_r + robot_r + 0.15) ** 2

    def predict_dyn_pos(self, real_time):
        """Predict dynamic obstacle centre at real_time seconds."""
        spd = self.dyn_info['speed']
        T_half = 1.0 / spd
        n = real_time * self.sim_freq
        phase = (n % (2 * T_half)) / T_half
        t_param = phase if phase <= 1.0 else 2.0 - phase
        s = np.array(self.dyn_info['path_start'])
        e = np.array(self.dyn_info['path_end'])
        return (1 - t_param) * s + t_param * e

    def _dyn_blocked(self, gx, gy, ts):
        dp = self.predict_dyn_pos(ts * self.dt)
        wx, wy = self.mapper.grid_to_world(gx, gy)
        dx, dy = wx - dp[0], wy - dp[1]
        return (dx * dx + dy * dy) < self.inflated_r_sq

    def _blocked(self, gx, gy, ts):
        W, H = self.mapper.width, self.mapper.height
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            return True
        if self.mapper.cspace_grid[gx, gy]:
            return True
        return self._dyn_blocked(gx, gy, ts)

    @staticmethod
    def heuristic(a, goal):
        return max(abs(a[0] - goal[0]), abs(a[1] - goal[1]))

    def plan(self, start_world, goal_world, max_time_steps=800):
        start = self.mapper.world_to_grid(*start_world)
        goal  = self.mapper.world_to_grid(*goal_world)
        if self.mapper.cspace_grid[start[0], start[1]] or \
           self.mapper.cspace_grid[goal[0], goal[1]]:
            return None, {'success': False, 'reason': 'static collision'}

        t0 = time.perf_counter()
        s0 = (start[0], start[1], 0)
        open_heap = [(self.heuristic(start, goal), 0, s0)]
        counter = 1
        g = {s0: 0.0}
        parent = {}
        closed = set()
        nodes = 0

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            nodes += 1
            gx, gy, ts = cur

            if (gx, gy) == goal:
                path = []
                st = cur
                while st in parent:
                    path.append(st)
                    st = parent[st]
                path.append(s0)
                path.reverse()
                timed = [(self.mapper.grid_to_world(s[0], s[1])[0],
                          self.mapper.grid_to_world(s[0], s[1])[1],
                          s[2] * self.dt) for s in path]
                xy = [(pt[0], pt[1]) for pt in timed]
                el = (time.perf_counter() - t0) * 1000
                return timed, {
                    'success': True, 'time_ms': el,
                    'path_length': euclidean_path_length(xy),
                    'nodes_expanded': nodes,
                    'arrival_time_s': timed[-1][2],
                }

            if ts >= max_time_steps:
                continue

            for dx, dy in self.MOVES:
                nx, ny, nt = gx + dx, gy + dy, ts + 1
                nb = (nx, ny, nt)
                if nb in closed or self._blocked(nx, ny, nt):
                    continue
                if dx != 0 and dy != 0:
                    if self._blocked(gx + dx, gy, ts) or \
                       self._blocked(gx, gy + dy, ts):
                        continue
                step_cost = 1.0 if (dx or dy) else 1.001
                tg = g[cur] + step_cost
                if tg < g.get(nb, float('inf')):
                    g[nb] = tg
                    parent[nb] = cur
                    f = tg + self.heuristic((nx, ny), goal)
                    heapq.heappush(open_heap, (f, counter, nb))
                    counter += 1

        el = (time.perf_counter() - t0) * 1000
        return None, {'success': False, 'time_ms': el,
                      'nodes_expanded': nodes, 'reason': 'time limit'}


# ====================================================================== #
#  Feasibility Check
# ====================================================================== #

def feasibility_check(mapper, start, goal):
    """
    Verify that a collision-free path exists between start and goal
    using A* on the C-space grid. Returns (feasible: bool, info: str).
    """
    planner = AStarPlanner(mapper)
    path, stats = planner.plan(start, goal, weight=1.0)
    if stats['success']:
        return True, f"Feasible (path length {stats['path_length']:.2f} m)"
    else:
        return False, f"Infeasible — {stats.get('reason', 'unknown')}"
