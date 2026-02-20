"""
main.py — Robot Planning Coursework Entry Point
================================================
Imports planning modules from planner.py and executes the full
navigation task: Phase 1 (C-space), Phase 2 (search), Phase 3 (dynamics).

"""

import env_factory
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from planner import (
    GridMapper,
    AStarPlanner,
    RRTConnectPlanner,
    PathSmoother,
    SpaceTimeAStar,
    euclidean_path_length,
    feasibility_check,
)


# ====================================================================== #
#  Visualisation helpers
# ====================================================================== #

def plot_workspace_vs_cspace(mapper, setup, student_id):
    """Task 1.3 — side-by-side workspace / C-space plot."""
    bounds = setup['map_bounds']
    fig, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(16, 7.5))

    # Left: Workspace
    ax_ws.imshow(mapper.grid.T, origin='lower', extent=bounds,
                 cmap=ListedColormap(['white', '#505050']), vmin=0, vmax=1)
    for obs in setup['static_obstacles']:
        ax_ws.add_patch(patches.Rectangle(
            (obs['pos'][0] - obs['extents'][0],
             obs['pos'][1] - obs['extents'][1]),
            2 * obs['extents'][0], 2 * obs['extents'][1],
            lw=1.2, ec='black', fc='#606060', alpha=0.85))
    sx, sy = setup['start']
    if setup['robot_type'] == "RECT":
        rw, rh = setup['robot_geometry']
        ax_ws.add_patch(patches.Rectangle(
            (sx - rw/2, sy - rh/2), rw, rh,
            lw=2, ec='red', fc='red', alpha=0.45, label='Robot'))
    else:
        pts = [(sx + v[0], sy + v[1]) for v in setup['robot_geometry']]
        ax_ws.add_patch(patches.Polygon(
            pts, closed=True, lw=2, ec='red', fc='red', alpha=0.45, label='Robot'))
    ax_ws.plot(*setup['start'], 'go', ms=10, zorder=5, label='Start')
    ax_ws.plot(*setup['goal'],  'r*', ms=14, zorder=5, label='Goal')
    ax_ws.set_title('Workspace', fontsize=14, weight='bold')
    ax_ws.set_xlabel('X (m)'); ax_ws.set_ylabel('Y (m)')
    ax_ws.legend(loc='upper left', fontsize=9)
    ax_ws.set_xlim(bounds[0], bounds[1]); ax_ws.set_ylim(bounds[2], bounds[3])
    ax_ws.set_aspect('equal'); ax_ws.grid(True, alpha=0.25)

    # RIGHT: C-Space
    comp = np.zeros_like(mapper.cspace_grid.T, dtype=int)
    comp[mapper.cspace_grid.T > 0] = 1
    comp[mapper.grid.T > 0] = 2
    ax_cs.imshow(comp, origin='lower', extent=bounds,
                 cmap=ListedColormap(['#FFFFFF', '#FFCCCC', '#505050']), vmin=0, vmax=2)
    ax_cs.plot([], [], 's', color='#FFCCCC', ms=10, label='C-space growth')
    ax_cs.plot([], [], 's', color='#505050', ms=10, label='Original obstacle')
    ax_cs.plot(*setup['start'], 'go', ms=10, zorder=5, label='Start (point)')
    ax_cs.plot(*setup['goal'],  'r*', ms=14, zorder=5, label='Goal (point)')
    ax_cs.set_title('Configuration Space', fontsize=14, weight='bold')
    ax_cs.set_xlabel('X (m)'); ax_cs.set_ylabel('Y (m)')
    ax_cs.legend(loc='upper left', fontsize=9)
    ax_cs.set_xlim(bounds[0], bounds[1]); ax_cs.set_ylim(bounds[2], bounds[3])
    ax_cs.set_aspect('equal'); ax_cs.grid(True, alpha=0.25)

    fig.suptitle(f'Phase 1: Workspace vs C-Space — Seed {student_id}',
                 fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig('phase1_workspace_cspace.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 1] Plot saved -> phase1_workspace_cspace.png")


def plot_phase2_paths(mapper, setup, results, student_id):
    """Task 2.3 — all Phase 2 paths on C-space."""
    bounds = setup['map_bounds']
    comp = np.zeros_like(mapper.cspace_grid.T, dtype=int)
    comp[mapper.cspace_grid.T > 0] = 1
    comp[mapper.grid.T > 0] = 2
    cmap_bg = ListedColormap(['#FFFFFF', '#FFCCCC', '#505050'])
    styles = {
        'A* (w=1.0)':  {'color': '#1f77b4', 'lw': 2.2, 'ls': '-'},
        'A* (w=1.5)':  {'color': '#ff7f0e', 'lw': 2.0, 'ls': '--'},
        'A* (w=5.0)':  {'color': '#2ca02c', 'lw': 2.0, 'ls': '-.'},
        'RRT-Connect': {'color': '#d62728', 'lw': 1.8, 'ls': '-'},
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, title in zip(axes, ['Grid Search (A*)', 'All Paths Overlay']):
        ax.imshow(comp, origin='lower', extent=bounds, cmap=cmap_bg, vmin=0, vmax=2)
        ax.plot(*setup['start'], 'go', ms=10, zorder=5, label='Start')
        ax.plot(*setup['goal'],  'r*', ms=14, zorder=5, label='Goal')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    for label in ['A* (w=1.0)', 'A* (w=1.5)', 'A* (w=5.0)']:
        res = results.get(label)
        if res and res['stats']['success']:
            path = res['path']
            sty = styles[label]
            axes[0].plot([pt[0] for pt in path], [pt[1] for pt in path],
                         color=sty['color'], lw=sty['lw'], ls=sty['ls'],
                         label=label, zorder=3)
    axes[0].legend(loc='upper left', fontsize=9)
    for label, sty in styles.items():
        res = results.get(label)
        if res and res['stats']['success']:
            path = res['path']
            axes[1].plot([pt[0] for pt in path], [pt[1] for pt in path],
                         color=sty['color'], lw=sty['lw'], ls=sty['ls'],
                         label=label, zorder=3)
    axes[1].legend(loc='upper left', fontsize=9)
    fig.suptitle(f'Phase 2: Search & Sampling — Seed {student_id}',
                 fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig('phase2_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 2] Plot saved -> phase2_paths.png")


def plot_phase2_expansion(mapper, setup, results, student_id):
    """
    Heatmap of A* expanded cells for each weight.
    Shows how higher weight explores fewer nodes but may deviate from optimal.
    """
    bounds = setup['map_bounds']
    weights = ['A* (w=1.0)', 'A* (w=1.5)', 'A* (w=5.0)']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, label in zip(axes, weights):
        res = results.get(label)
        # C-space background
        comp = np.zeros_like(mapper.cspace_grid.T, dtype=float)
        comp[mapper.cspace_grid.T > 0] = 0.3
        comp[mapper.grid.T > 0] = 1.0

        # Overlay expanded cells
        if res and res['stats']['success']:
            expanded = res['stats'].get('expanded_cells', set())
            exp_grid = np.zeros((mapper.width, mapper.height), dtype=float)
            for (gx, gy) in expanded:
                if 0 <= gx < mapper.width and 0 <= gy < mapper.height:
                    exp_grid[gx, gy] = 1.0
            # Show expansion as coloured overlay
            ax.imshow(comp, origin='lower', extent=bounds,
                      cmap='Greys', vmin=0, vmax=1, alpha=0.4)
            ax.imshow(np.ma.masked_where(exp_grid.T == 0, exp_grid.T),
                      origin='lower', extent=bounds,
                      cmap='YlOrRd', vmin=0, vmax=1, alpha=0.6)
            # Path
            path = res['path']
            ax.plot([p[0] for p in path], [p[1] for p in path],
                    'b-', lw=2.5, zorder=4, label='Path')
            s = res['stats']
            ax.set_title(f"{label}\n"
                         f"nodes={s['nodes_expanded']}  "
                         f"length={s['path_length']:.2f} m  "
                         f"time={s['time_ms']:.1f} ms",
                         fontsize=11, weight='bold')
        else:
            ax.imshow(comp, origin='lower', extent=bounds, cmap='Greys')
            ax.set_title(f"{label}\nFAILED", fontsize=11, weight='bold')

        ax.plot(*setup['start'], 'go', ms=10, zorder=5)
        ax.plot(*setup['goal'],  'r*', ms=14, zorder=5)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.15)

    fig.suptitle(f'A* Node Expansion Comparison — Seed {student_id}',
                 fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig('phase2_expansion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 2] Expansion plot saved -> phase2_expansion.png")


def plot_phase2_bars(results, student_id):
    """Bar chart comparing Time, Path Length, and Memory across algorithms."""
    labels = []
    times = []
    lengths = []
    memory = []
    mem_labels_list = []

    for label in ['A* (w=1.0)', 'A* (w=1.5)', 'A* (w=5.0)', 'RRT-Connect']:
        res = results.get(label)
        if not res or not res['stats']['success']:
            continue
        s = res['stats']
        labels.append(label)
        times.append(s['time_ms'])
        lengths.append(s['path_length'])
        if 'nodes_expanded' in s:
            memory.append(s['nodes_expanded'])
            mem_labels_list.append('Nodes expanded')
        else:
            memory.append(s.get('tree_size', s.get('vertices_sampled', 0)))
            mem_labels_list.append('Tree nodes')

    # Adding RRT average if available
    rrt_runs = results.get('_rrt_runs', [])
    ok_rrt = [r for r in rrt_runs if r['stats']['success']]
    if ok_rrt:
        labels.append('RRT (avg 5)')
        times.append(np.mean([r['stats']['time_ms'] for r in ok_rrt]))
        lengths.append(np.mean([r['stats']['path_length'] for r in ok_rrt]))
        memory.append(np.mean([r['stats']['tree_size'] for r in ok_rrt]))
        mem_labels_list.append('Tree nodes')

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Time
    bars = axes[0].bar(labels, times, color=colours[:len(labels)], edgecolor='black', lw=0.5)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time', fontsize=12, weight='bold')
    for bar, val in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    axes[0].tick_params(axis='x', rotation=20)

    # Path length
    bars = axes[1].bar(labels, lengths, color=colours[:len(labels)], edgecolor='black', lw=0.5)
    axes[1].set_ylabel('Path Length (m)')
    axes[1].set_title('Path Length (Optimality)', fontsize=12, weight='bold')
    for bar, val in zip(bars, lengths):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    axes[1].tick_params(axis='x', rotation=20)

    # Memory
    bars = axes[2].bar(labels, memory, color=colours[:len(labels)], edgecolor='black', lw=0.5)
    axes[2].set_ylabel('Count')
    axes[2].set_title('Memory (Nodes / Vertices)', fontsize=12, weight='bold')
    for bar, val in zip(bars, memory):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{int(val)}', ha='center', va='bottom', fontsize=9)
    axes[2].tick_params(axis='x', rotation=20)

    fig.suptitle(f'Phase 2: Algorithm Comparison — Seed {student_id}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('phase2_comparison_bars.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 2] Bar chart saved -> phase2_comparison_bars.png")


def plot_phase2_rrt_variability(results, student_id):
    """Show path length and time variability across the 5 RRT runs."""
    rrt_runs = results.get('_rrt_runs', [])
    ok_runs = [r for r in rrt_runs if r['stats']['success']]
    if not ok_runs:
        print("[Phase 2] No successful RRT runs — skipping variability plot.")
        return

    run_ids = list(range(1, len(ok_runs) + 1))
    times   = [r['stats']['time_ms'] for r in ok_runs]
    lengths = [r['stats']['path_length'] for r in ok_runs]
    trees   = [r['stats']['tree_size'] for r in ok_runs]

    # A* (w=1.0) baseline for reference
    astar_res = results.get('A* (w=1.0)')
    astar_len = astar_res['stats']['path_length'] if astar_res and astar_res['stats']['success'] else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(run_ids, times, color='#d62728', edgecolor='black', lw=0.5)
    axes[0].set_xlabel('RRT Run'); axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time', fontsize=12, weight='bold')
    axes[0].set_xticks(run_ids)

    axes[1].bar(run_ids, lengths, color='#d62728', edgecolor='black', lw=0.5)
    if astar_len is not None:
        axes[1].axhline(astar_len, color='#1f77b4', ls='--', lw=2,
                         label=f'A* optimal ({astar_len:.2f} m)')
        axes[1].legend(fontsize=9)
    axes[1].set_xlabel('RRT Run'); axes[1].set_ylabel('Path Length (m)')
    axes[1].set_title('Path Length', fontsize=12, weight='bold')
    axes[1].set_xticks(run_ids)

    axes[2].bar(run_ids, trees, color='#d62728', edgecolor='black', lw=0.5)
    axes[2].set_xlabel('RRT Run'); axes[2].set_ylabel('Tree Nodes')
    axes[2].set_title('Tree Size (Memory)', fontsize=12, weight='bold')
    axes[2].set_xticks(run_ids)

    fig.suptitle(f'RRT-Connect Variability (5 Runs) — Seed {student_id}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('phase2_rrt_variability.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 2] RRT variability plot saved -> phase2_rrt_variability.png")


def plot_phase3(mapper, setup, raw_path, smooth_path,
                timed_path, sta_stats, student_id):
    """Phase 3 figure: smoothing + Space-Time A*."""
    bounds = setup['map_bounds']

    def draw_bg(ax):
        comp = np.zeros_like(mapper.cspace_grid.T, dtype=int)
        comp[mapper.cspace_grid.T > 0] = 1
        comp[mapper.grid.T > 0] = 2
        ax.imshow(comp, origin='lower', extent=bounds,
                  cmap=ListedColormap(['#FFFFFF', '#FFCCCC', '#505050']), vmin=0, vmax=2)
        ax.plot(*setup['start'], 'go', ms=10, zorder=5, label='Start')
        ax.plot(*setup['goal'],  'r*', ms=14, zorder=5, label='Goal')
        ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    # LEFT: smoothing
    draw_bg(ax1)
    ax1.plot([pt[0] for pt in raw_path], [pt[1] for pt in raw_path],
             '-', color='#888888', lw=1.2, alpha=0.7,
             label=f'Raw A* ({len(raw_path)} pts)')
    ax1.plot([pt[0] for pt in smooth_path], [pt[1] for pt in smooth_path],
             '-', color='#1f77b4', lw=2.2,
             label=f'Smoothed ({len(smooth_path)} pts)')
    ax1.set_title('Task 3.1: Path Smoothing', fontsize=14, weight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    # RIGHT: Space-Time A*
    draw_bg(ax2)
    dyn = setup['dynamic_obstacle']
    ax2.plot([dyn['path_start'][0], dyn['path_end'][0]],
             [dyn['path_start'][1], dyn['path_end'][1]],
             'b--', lw=1.5, alpha=0.4, label='Obstacle patrol')
    if timed_path is not None:
        tx = [pt[0] for pt in timed_path]
        ty = [pt[1] for pt in timed_path]
        tt = [pt[2] for pt in timed_path]
        for k in range(1, len(tx)):
            ax2.plot(tx[k-1:k+1], ty[k-1:k+1], '-',
                     color=plt.cm.plasma(tt[k] / max(tt[-1], 1e-6)),
                     lw=2.0, zorder=3)
        sta_vis = SpaceTimeAStar(mapper, dyn, setup['robot_type'],
                                 setup['robot_geometry'])
        n_snap = 6
        indices = np.linspace(0, len(timed_path)-1, n_snap, dtype=int)
        for idx in indices:
            dp = sta_vis.predict_dyn_pos(timed_path[idx][2])
            ax2.add_patch(plt.Circle(dp, dyn['radius'], color='blue',
                                     alpha=0.2, zorder=2))
            ax2.annotate(f't={timed_path[idx][2]:.1f}s', dp, fontsize=7,
                         ha='center', color='blue', alpha=0.6)
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, tt[-1]))
        sm.set_array([])
        fig.colorbar(sm, ax=ax2, fraction=0.04, pad=0.02).set_label('Time (s)')
        info = (f"length={sta_stats['path_length']:.2f} m  "
                f"arrival={sta_stats['arrival_time_s']:.1f} s  "
                f"nodes={sta_stats['nodes_expanded']}")
    else:
        info = "FAILED"
    ax2.set_title('Task 3.2: Space-Time A*', fontsize=14, weight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.text(0.02, 0.02, info, transform=ax2.transAxes, fontsize=8,
             va='bottom', bbox=dict(fc='white', alpha=0.8))

    fig.suptitle(f'Phase 3: Dynamics & Refinement — Seed {student_id}',
                 fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig('phase3_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 3] Plot saved -> phase3_dynamics.png")


def plot_phase3_smoothing_detail(raw_path, smooth_path, student_id):
    """
    Detailed smoothing analysis:
      Left:  zoomed-in overlay of raw vs smooth near a turning point
      Right: curvature profile along the smooth path (proves C2 continuity)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Panel 1: full path comparison with control points 
    raw_arr = np.array(raw_path)
    sm_arr  = np.array(smooth_path)

    axes[0].plot(raw_arr[:, 0], raw_arr[:, 1], 'o-', color='#888888',
                 ms=1.5, lw=0.8, alpha=0.6, label=f'Raw A* ({len(raw_path)} pts)')
    axes[0].plot(sm_arr[:, 0], sm_arr[:, 1], '-', color='#1f77b4',
                 lw=2.2, label=f'Smooth ({len(smooth_path)} pts)')
    axes[0].plot(raw_arr[0, 0], raw_arr[0, 1], 'go', ms=10, zorder=5)
    axes[0].plot(raw_arr[-1, 0], raw_arr[-1, 1], 'r*', ms=14, zorder=5)
    axes[0].set_title('Full Path Comparison', fontsize=12, weight='bold')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_aspect('equal'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Panel 2: first / second derivative magnitude 
    diffs = np.diff(sm_arr, axis=0)
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    arc = np.concatenate([[0], np.cumsum(seg_len)])
    arc_norm = arc / arc[-1] if arc[-1] > 0 else arc

    dx = np.gradient(sm_arr[:, 0], arc_norm)
    dy = np.gradient(sm_arr[:, 1], arc_norm)
    ddx = np.gradient(dx, arc_norm)
    ddy = np.gradient(dy, arc_norm)
    speed = np.sqrt(dx**2 + dy**2)
    accel = np.sqrt(ddx**2 + ddy**2)

    axes[1].plot(arc_norm, speed, '-', color='#1f77b4', lw=1.5,
                 label="||path'(s)||  (1st deriv)")
    axes[1].plot(arc_norm, accel, '-', color='#d62728', lw=1.5, alpha=0.8,
                 label="||path''(s)|| (2nd deriv)")
    axes[1].set_title("Derivative Continuity", fontsize=12, weight='bold')
    axes[1].set_xlabel('Normalised arc length s')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)

    # Panel 3: curvature κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2) 
    numer = np.abs(dx * ddy - dy * ddx)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)
    curvature = numer / denom

    axes[2].plot(arc_norm, curvature, '-', color='#2ca02c', lw=1.5)
    axes[2].set_title('Curvature Profile (C2 proof)', fontsize=12, weight='bold')
    axes[2].set_xlabel('Normalised arc length s')
    axes[2].set_ylabel('Curvature κ (1/m)')
    axes[2].grid(True, alpha=0.2)
    axes[2].text(0.98, 0.95, 'Continuous κ → C2 path',
                 transform=axes[2].transAxes, ha='right', va='top',
                 fontsize=9, style='italic',
                 bbox=dict(fc='lightyellow', alpha=0.8))

    fig.suptitle(f'Task 3.1: Smoothing Analysis — Seed {student_id}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('phase3_smoothing_detail.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 3] Smoothing detail saved -> phase3_smoothing_detail.png")


def plot_phase3_distance_profile(mapper, setup, timed_path, student_id):
    """
    Minimum distance between robot and dynamic obstacle at each time step.
    Proves the robot never collides.
    """
    if timed_path is None:
        print("[Phase 3] No timed path — skipping distance profile.")
        return

    dyn = setup['dynamic_obstacle']
    sta = SpaceTimeAStar(mapper, dyn, setup['robot_type'], setup['robot_geometry'])

    times = [pt[2] for pt in timed_path]
    dists = []
    dyn_xs, dyn_ys = [], []
    rob_xs, rob_ys = [], []

    for x, y, t in timed_path:
        dp = sta.predict_dyn_pos(t)
        d = np.hypot(x - dp[0], y - dp[1])
        dists.append(d)
        dyn_xs.append(dp[0]); dyn_ys.append(dp[1])
        rob_xs.append(x); rob_ys.append(y)

    # Safety radius = dyn_radius + robot circumscribed radius
    dyn_r = dyn['radius']
    if setup['robot_type'] == "RECT":
        rob_r = np.hypot(setup['robot_geometry'][0]/2, setup['robot_geometry'][1]/2)
    else:
        rob_r = max(np.hypot(v[0], v[1]) for v in setup['robot_geometry'])
    safety_r = dyn_r + rob_r

    min_dist = min(dists)
    min_idx = dists.index(min_dist)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: distance over time
    axes[0].plot(times, dists, '-', color='#1f77b4', lw=1.8, label='Distance to dyn. obs.')
    axes[0].axhline(safety_r, color='red', ls='--', lw=1.5,
                     label=f'Collision radius ({safety_r:.2f} m)')
    axes[0].axhline(safety_r + 0.15, color='orange', ls=':', lw=1.2,
                     label=f'Safety margin ({safety_r+0.15:.2f} m)')
    axes[0].plot(times[min_idx], min_dist, 'rv', ms=12, zorder=5,
                 label=f'Min dist = {min_dist:.2f} m (t={times[min_idx]:.1f}s)')
    axes[0].fill_between(times, 0, safety_r, alpha=0.1, color='red')
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Distance (m)')
    axes[0].set_title('Robot–Obstacle Distance', fontsize=12, weight='bold')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(bottom=0)

    # Right: spatial view with closest-approach highlight
    bounds = setup['map_bounds']
    comp = np.zeros_like(mapper.cspace_grid.T, dtype=int)
    comp[mapper.cspace_grid.T > 0] = 1
    comp[mapper.grid.T > 0] = 2
    axes[1].imshow(comp, origin='lower', extent=bounds,
                   cmap=ListedColormap(['#FFFFFF', '#FFCCCC', '#505050']),
                   vmin=0, vmax=2, alpha=0.5)
    axes[1].plot(rob_xs, rob_ys, '-', color='#1f77b4', lw=1.5, alpha=0.7, label='Robot')
    axes[1].plot(dyn_xs, dyn_ys, '-', color='blue', lw=1.5, alpha=0.5, label='Dyn. obstacle')
    # Closest approach
    axes[1].plot(rob_xs[min_idx], rob_ys[min_idx], 'rv', ms=12, zorder=5)
    axes[1].plot(dyn_xs[min_idx], dyn_ys[min_idx], 'b^', ms=12, zorder=5)
    axes[1].plot([rob_xs[min_idx], dyn_xs[min_idx]],
                 [rob_ys[min_idx], dyn_ys[min_idx]],
                 'r-', lw=2, zorder=4, label=f'Closest approach ({min_dist:.2f} m)')
    axes[1].add_patch(plt.Circle((dyn_xs[min_idx], dyn_ys[min_idx]),
                                  safety_r, fc='none', ec='red', ls='--', lw=1.5))
    axes[1].plot(*setup['start'], 'go', ms=10, zorder=5)
    axes[1].plot(*setup['goal'],  'r*', ms=14, zorder=5)
    axes[1].set_xlim(bounds[0], bounds[1]); axes[1].set_ylim(bounds[2], bounds[3])
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Closest Approach (Spatial)', fontsize=12, weight='bold')
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f'Task 3.2: Dynamic Obstacle Safety — Seed {student_id}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('phase3_distance_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 3] Distance profile saved -> phase3_distance_profile.png")


def plot_phase3_spacetime(mapper, setup, timed_path, student_id):
    """
    Space-time trajectory diagram: x(t) and y(t) for robot and obstacle.
    Shows how robot waits or re-routes to avoid crossing the obstacle path.
    """
    if timed_path is None:
        print("[Phase 3] No timed path — skipping space-time plot.")
        return

    dyn = setup['dynamic_obstacle']
    sta = SpaceTimeAStar(mapper, dyn, setup['robot_type'], setup['robot_geometry'])

    rob_t = [pt[2] for pt in timed_path]
    rob_x = [pt[0] for pt in timed_path]
    rob_y = [pt[1] for pt in timed_path]

    t_max = rob_t[-1]
    t_dense = np.linspace(0, t_max, 500)
    obs_x, obs_y = [], []
    for t in t_dense:
        dp = sta.predict_dyn_pos(t)
        obs_x.append(dp[0]); obs_y.append(dp[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # X(t)
    axes[0].plot(rob_t, rob_x, '-', color='#1f77b4', lw=2, label='Robot x(t)')
    axes[0].plot(t_dense, obs_x, '-', color='blue', lw=1.5, alpha=0.5, label='Obstacle x(t)')
    axes[0].axhline(setup['start'][0], color='green', ls=':', alpha=0.4, label='Start x')
    axes[0].axhline(setup['goal'][0], color='red', ls=':', alpha=0.4, label='Goal x')
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('X (m)')
    axes[0].set_title('X coordinate over time', fontsize=12, weight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.2)

    # Y(t)
    axes[1].plot(rob_t, rob_y, '-', color='#d62728', lw=2, label='Robot y(t)')
    axes[1].plot(t_dense, obs_y, '-', color='blue', lw=1.5, alpha=0.5, label='Obstacle y(t)')
    axes[1].axhline(setup['start'][1], color='green', ls=':', alpha=0.4, label='Start y')
    axes[1].axhline(setup['goal'][1], color='red', ls=':', alpha=0.4, label='Goal y')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Y coordinate over time', fontsize=12, weight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.2)

    fig.suptitle(f'Task 3.2: Space-Time Trajectories — Seed {student_id}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('phase3_spacetime.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 3] Space-time plot saved -> phase3_spacetime.png")


def plot_phase3_static_vs_dynamic(mapper, setup, static_path, timed_path,
                                  student_id):
    """
    Overlay the static A* path and the Space-Time A* path on the same map
    to show how the robot deviates to avoid the moving obstacle.
    """
    if timed_path is None:
        print("[Phase 3] No timed path — skipping static vs dynamic plot.")
        return

    bounds = setup['map_bounds']
    dyn = setup['dynamic_obstacle']

    fig, ax = plt.subplots(figsize=(10, 9))
    comp = np.zeros_like(mapper.cspace_grid.T, dtype=int)
    comp[mapper.cspace_grid.T > 0] = 1
    comp[mapper.grid.T > 0] = 2
    ax.imshow(comp, origin='lower', extent=bounds,
              cmap=ListedColormap(['#FFFFFF', '#FFCCCC', '#505050']),
              vmin=0, vmax=2)

    # Static path
    sp = np.array(static_path)
    ax.plot(sp[:, 0], sp[:, 1], '-', color='#888888', lw=2.5, alpha=0.7,
            label=f'Static A* ({euclidean_path_length(static_path):.2f} m)')

    # Dynamic path coloured by time
    tx = [pt[0] for pt in timed_path]
    ty = [pt[1] for pt in timed_path]
    tt = [pt[2] for pt in timed_path]
    for k in range(1, len(tx)):
        ax.plot(tx[k-1:k+1], ty[k-1:k+1], '-',
                color=plt.cm.plasma(tt[k] / max(tt[-1], 1e-6)),
                lw=2.5, zorder=3)

    # Obstacle patrol line
    ax.plot([dyn['path_start'][0], dyn['path_end'][0]],
            [dyn['path_start'][1], dyn['path_end'][1]],
            'b--', lw=2, alpha=0.4, label='Obstacle patrol line')

    # Obstacle snapshots
    sta = SpaceTimeAStar(mapper, dyn, setup['robot_type'], setup['robot_geometry'])
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (len(timed_path) - 1)), len(timed_path) - 1)
        rt = timed_path[idx][2]
        dp = sta.predict_dyn_pos(rt)
        circ = plt.Circle(dp, dyn['radius'], fc='blue', ec='darkblue',
                           alpha=0.15, lw=1.2, zorder=2)
        ax.add_patch(circ)
        ax.annotate(f't={rt:.1f}s', dp, fontsize=8, ha='center',
                    color='darkblue', alpha=0.7, weight='bold')

    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, tt[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label('Time (s)')

    dyn_len = euclidean_path_length([(x, y) for x, y, _ in timed_path])
    # Legend proxy for the coloured line
    ax.plot([], [], '-', color=plt.cm.plasma(0.5), lw=2.5,
            label=f'Space-Time A* ({dyn_len:.2f} m)')

    ax.plot(*setup['start'], 'go', ms=12, zorder=5, label='Start')
    ax.plot(*setup['goal'],  'r*', ms=16, zorder=5, label='Goal')
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(f'Static A* vs Space-Time A* — Seed {student_id}',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('phase3_static_vs_dynamic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Phase 3] Static vs dynamic saved -> phase3_static_vs_dynamic.png")


# ====================================================================== #
#  Phase orchestration
# ====================================================================== #

def run_phase2(mapper, setup, student_id):
    """Run all Phase 2 planners, print comparison table, return results."""
    start = tuple(setup['start'])
    goal  = tuple(setup['goal'])
    results = {}

    astar = AStarPlanner(mapper)
    for w in [1.0, 1.5, 5.0]:
        label = f"A* (w={w})"
        print(f"  Running {label} ...")
        path, stats = astar.plan(start, goal, weight=w)
        results[label] = {'path': path, 'stats': stats}
        if stats['success']:
            print(f"    time={stats['time_ms']:.1f} ms  "
                  f"length={stats['path_length']:.2f} m  "
                  f"nodes={stats['nodes_expanded']}")
        else:
            print(f"    FAILED - {stats.get('reason')}")

    rrt = RRTConnectPlanner(mapper, step_size=0.5, max_iter=10000)
    n_runs = 5
    rrt_runs = []
    print(f"  Running RRT-Connect ({n_runs} runs, step=0.5 m) ...")
    for ri in range(n_runs):
        np.random.seed(student_id * 10 + ri)
        path, stats = rrt.plan(start, goal)
        rrt_runs.append({'path': path, 'stats': stats})
        ok = "OK" if stats['success'] else "FAIL"
        print(f"    run {ri+1}: {ok}  time={stats['time_ms']:.1f} ms  "
              f"length={stats.get('path_length', float('nan')):.2f} m  "
              f"vertices={stats.get('vertices_sampled',0)}  "
              f"tree={stats.get('tree_size',0)}")

    ok_runs = [r for r in rrt_runs if r['stats']['success']]
    best_rrt = min(ok_runs, key=lambda r: r['stats']['path_length']) if ok_runs else rrt_runs[0]
    results['RRT-Connect'] = best_rrt

    # Comparison table
    print("\n" + "=" * 85)
    print(f"  {'Algorithm':<20} {'Time (ms)':>10} {'Path Length (m)':>16} "
          f"{'Nodes / Verts':>15} {'Memory metric':>14}")
    print("  " + "-" * 81)
    for label, res in results.items():
        s = res['stats']
        if not s['success']:
            print(f"  {label:<20} {'FAILED':>10}"); continue
        mem_val = s.get('nodes_expanded', s.get('tree_size', s.get('vertices_sampled', 0)))
        mem_lbl = 'expanded' if 'nodes_expanded' in s else 'tree nodes'
        print(f"  {label:<20} {s['time_ms']:>10.1f} {s['path_length']:>16.2f} "
              f"{mem_val:>15}   ({mem_lbl})")
    if ok_runs:
        print(f"  {'RRT-Connect (avg)':<20} "
              f"{np.mean([r['stats']['time_ms'] for r in ok_runs]):>10.1f} "
              f"{np.mean([r['stats']['path_length'] for r in ok_runs]):>16.2f} "
              f"{np.mean([r['stats']['tree_size'] for r in ok_runs]):>15.0f}   (tree nodes)")
    print("  " + "-" * 81)
    print("=" * 85)

    results['_rrt_runs'] = rrt_runs
    return results


def run_phase3(mapper, setup, phase2_results, student_id):
    """Run all Phase 3 tasks."""
    astar_res = phase2_results.get('A* (w=1.0)')
    if not astar_res or not astar_res['stats']['success']:
        print("  No A* path available — Phase 3 cannot proceed.")
        return None
    raw_path = astar_res['path']

    # Task 3.1: Smoothing
    print("\n[Task 3.1] Smoothing A* path ...")
    smoother = PathSmoother(mapper)
    smooth = smoother.smooth(raw_path, num_points=300)
    print(f"  Raw   : {len(raw_path):>5} waypoints, "
          f"{euclidean_path_length(raw_path):.2f} m")
    print(f"  Smooth: {len(smooth):>5} waypoints, "
          f"{euclidean_path_length(smooth):.2f} m  (C2 cubic spline)")

    # Task 3.2: Space-Time A*
    print("\n[Task 3.2] Planning with Space-Time A* ...")
    sta = SpaceTimeAStar(mapper, setup['dynamic_obstacle'],
                         setup['robot_type'], setup['robot_geometry'],
                         robot_speed=1.0)
    timed_path, sta_stats = sta.plan(
        tuple(setup['start']), tuple(setup['goal']), max_time_steps=800)
    if sta_stats['success']:
        print(f"  Path found: length={sta_stats['path_length']:.2f} m  "
              f"arrival={sta_stats['arrival_time_s']:.1f} s  "
              f"nodes={sta_stats['nodes_expanded']}  "
              f"time={sta_stats['time_ms']:.0f} ms")
    else:
        print(f"  FAILED - {sta_stats.get('reason')}")

    # Visualisation
    print("\n[Phase 3] Generating plots ...")
    plot_phase3(mapper, setup, raw_path, smooth,
                timed_path, sta_stats if sta_stats['success'] else {},
                student_id)
    plot_phase3_smoothing_detail(raw_path, smooth, student_id)
    plot_phase3_distance_profile(mapper, setup, timed_path, student_id)
    plot_phase3_spacetime(mapper, setup, timed_path, student_id)
    plot_phase3_static_vs_dynamic(mapper, setup, raw_path, timed_path,
                                  student_id)

    # Task 3.3: PyBullet execution
    if timed_path is not None:
        print("\n[Task 3.3] Launching PyBullet execution ...")
        execute_in_pybullet(student_id, timed_path)
    else:
        print("\n[Task 3.3] No dynamic path — skipping execution.")

    return {'smooth_path': smooth, 'timed_path': timed_path,
            'sta_stats': sta_stats}


def execute_in_pybullet(student_id, timed_path, sim_freq=240):
    """Open PyBullet GUI and drive the robot along the timed path."""
    env = env_factory.RandomizedWarehouse(seed=student_id, mode=p.GUI)
    env.activate_dynamic_obstacle()
    path_idx = 0
    sim_step = 0
    print("  [PyBullet] Executing ... (close window or Ctrl-C to stop)")
    try:
        while path_idx < len(timed_path):
            env.update_simulation()
            sim_step += 1
            real_t = sim_step / sim_freq
            x, y, t_target = timed_path[path_idx]
            if real_t >= t_target:
                p.resetBasePositionAndOrientation(
                    env.robot_id, [x, y, 0.1], [0, 0, 0, 1])
                path_idx += 1
            time.sleep(1.0 / sim_freq)
        for _ in range(sim_freq * 2):
            env.update_simulation()
            time.sleep(1.0 / sim_freq)
    except KeyboardInterrupt:
        pass
    finally:
        print("  [PyBullet] Done.")


# ====================================================================== #
#  MAIN
# ====================================================================== #

def main():

    STUDENT_ID = 4331

    # Initialising environment (headless for computation)
    env   = env_factory.RandomizedWarehouse(seed=STUDENT_ID, mode=p.DIRECT)
    setup = env.get_problem_setup()

    print("=" * 60)
    print(f"  Environment loaded — Seed: {STUDENT_ID}")
    print(f"  Version hash : {setup['version_hash']}")
    print(f"  Robot type   : {setup['robot_type']}")
    print(f"  Robot geom   : {setup['robot_geometry']}")
    print(f"  Start        : ({setup['start'][0]:.2f}, {setup['start'][1]:.2f})")
    print(f"  Goal         : ({setup['goal'][0]:.2f}, {setup['goal'][1]:.2f})")
    print(f"  # obstacles  : {len(setup['static_obstacles'])}")
    print("=" * 60)

    # ============================================================
    # PHASE 1 — The Geometry of Planning
    # ============================================================
    mapper = GridMapper(setup, resolution=0.1)
    mapper.fill_obstacles(setup['static_obstacles'])
    print(f"\n[Task 1.2] Workspace grid: {mapper.width} x {mapper.height} cells "
          f"(res {mapper.res} m)")

    print("\n[Task 1.1] Computing C-space obstacles ...")
    mapper.compute_cspace(setup['robot_type'], setup['robot_geometry'])
    free = int(np.sum(mapper.cspace_grid == 0))
    total = mapper.width * mapper.height
    print(f"  Free: {free}/{total} ({100*free/total:.1f}%)")

    start_ok = not mapper.check_collision(*setup['start'])
    goal_ok  = not mapper.check_collision(*setup['goal'])
    print(f"\n[Task 1.2] Start: {'FREE' if start_ok else 'BLOCKED'}  |  "
          f"Goal: {'FREE' if goal_ok else 'BLOCKED'}")

    # Feasibility check
    feasible, info = feasibility_check(
        mapper, tuple(setup['start']), tuple(setup['goal']))
    print(f"[Feasibility] {info}")
    if not feasible:
        print("  Problem is infeasible — exiting.")
        return

    print("\n[Task 1.3] Generating workspace vs C-space plot ...")
    plot_workspace_vs_cspace(mapper, setup, STUDENT_ID)
    print("\n[Phase 1 complete]")

    # ============================================================
    # PHASE 2 — Search and Sampling
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Search and Sampling")
    print("=" * 60)
    results = run_phase2(mapper, setup, STUDENT_ID)
    print("\n[Task 2.3] Generating comparison plots ...")
    plot_phase2_paths(mapper, setup, results, STUDENT_ID)
    plot_phase2_expansion(mapper, setup, results, STUDENT_ID)
    plot_phase2_bars(results, STUDENT_ID)
    plot_phase2_rrt_variability(results, STUDENT_ID)
    print("\n[Phase 2 complete]")

    # ============================================================
    # PHASE 3 — Dynamics and Refinement
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 3: Dynamics and Refinement")
    print("=" * 60)
    run_phase3(mapper, setup, results, STUDENT_ID)

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print(f"  Version hash: {setup['version_hash']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
