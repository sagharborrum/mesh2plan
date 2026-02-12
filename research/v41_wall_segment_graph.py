#!/usr/bin/env python3
"""
mesh2plan v41 - Wall Segment Graph

FUNDAMENTALLY DIFFERENT from v27-v40 (all raster-based).

Vector-first approach: extract wall geometry directly from mesh faces.
1. Identify wall faces by face normal (near-horizontal normals = vertical surfaces = walls)
2. Project wall face edges onto XZ plane → line segments
3. Cluster segments by angle + proximity → wall lines
4. Snap to dominant angles (like v40 but on vector data)
5. Build planar subdivision graph from wall lines
6. Find rooms as faces of the planar graph (minimal cycles)
7. Classify rooms by area

Key advantage: works directly with mesh geometry, not lossy density images.
No watershed, no contours, no raster-to-vector conversion artifacts.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import math
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict
import shutil


def extract_wall_faces(mesh, normal_thresh=0.7):
    """
    Wall faces have near-horizontal normals (small Y component).
    normal_thresh: max abs(normal.y) to be considered a wall.
    """
    normals = mesh.face_normals
    y_component = np.abs(normals[:, 1])
    wall_mask = y_component < normal_thresh
    print(f"  Total faces: {len(mesh.faces)}, Wall faces: {wall_mask.sum()} ({wall_mask.mean()*100:.1f}%)")
    return wall_mask


def project_wall_edges(mesh, wall_mask):
    """
    Project edges of wall faces onto XZ plane.
    Returns array of line segments [(x1,z1,x2,z2), ...].
    """
    verts = mesh.vertices
    wall_faces = mesh.faces[wall_mask]
    
    segments = set()
    for face in wall_faces:
        for k in range(3):
            i, j = face[k], face[(k + 1) % 3]
            edge = (min(i, j), max(i, j))
            segments.add(edge)
    
    lines = []
    for i, j in segments:
        x1, z1 = verts[i, 0], verts[i, 2]
        x2, z2 = verts[j, 0], verts[j, 2]
        length = math.sqrt((x2 - x1)**2 + (z2 - z1)**2)
        if length > 0.02:  # Skip tiny edges
            lines.append((x1, z1, x2, z2, length))
    
    print(f"  Wall edge segments: {len(lines)}")
    return np.array(lines)


def cluster_segments_by_angle(segments, angle_bin_deg=5):
    """
    Group segments by their angle (mod 180°).
    Returns dict: angle_bin → list of segments.
    """
    bins = defaultdict(list)
    for seg in segments:
        x1, z1, x2, z2, length = seg
        angle = math.atan2(z2 - z1, x2 - x1) % math.pi
        bin_idx = int(angle / math.pi * (180 / angle_bin_deg))
        bins[bin_idx].append(seg)
    return bins


def find_dominant_angles(segments, n_angles=4):
    """
    Find dominant wall angles weighted by segment length.
    """
    angle_weights = defaultdict(float)
    bin_deg = 5
    n_bins = 180 // bin_deg
    
    for seg in segments:
        x1, z1, x2, z2, length = seg
        angle = math.atan2(z2 - z1, x2 - x1) % math.pi
        bin_idx = int(angle / math.pi * n_bins) % n_bins
        angle_weights[bin_idx] += length
    
    # Smooth
    weights = np.zeros(n_bins)
    for b, w in angle_weights.items():
        weights[b] = w
    
    # Circular smoothing
    ext = np.concatenate([weights[-3:], weights, weights[:3]])
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    smoothed = np.convolve(ext, kernel, mode='same')[3:-3]
    
    # Find peaks
    peaks = []
    for i in range(n_bins):
        prev = (i - 1) % n_bins
        nxt = (i + 1) % n_bins
        if smoothed[i] > smoothed[prev] and smoothed[i] > smoothed[nxt] and smoothed[i] > 0:
            angle = (i + 0.5) * bin_deg / 180.0 * math.pi
            peaks.append((smoothed[i], angle))
    
    peaks.sort(reverse=True)
    
    # Deduplicate (merge within 10°)
    dominant = []
    for w, a in peaks:
        too_close = False
        for existing in dominant:
            diff = abs(a - existing) % math.pi
            if diff > math.pi / 2:
                diff = math.pi - diff
            if diff < math.radians(10):
                too_close = True
                break
        if not too_close:
            dominant.append(a)
        if len(dominant) >= n_angles:
            break
    
    dominant.sort()
    print(f"  Dominant angles: {[f'{math.degrees(a):.1f}°' for a in dominant]}")
    return dominant


def merge_collinear_segments(segments, angles, angle_thresh_deg=10, dist_thresh=0.15):
    """
    Merge nearby segments that share the same dominant angle.
    Project segments onto their dominant angle direction, merge overlapping projections.
    """
    angle_thresh = math.radians(angle_thresh_deg)
    
    # Assign each segment to nearest dominant angle
    assigned = defaultdict(list)  # angle_idx -> list of (proj_start, proj_end, perp_offset, seg)
    
    for seg in segments:
        x1, z1, x2, z2, length = seg
        seg_angle = math.atan2(z2 - z1, x2 - x1) % math.pi
        
        best_idx = None
        best_diff = float('inf')
        for i, a in enumerate(angles):
            diff = abs(seg_angle - a) % math.pi
            if diff > math.pi / 2:
                diff = math.pi - diff
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        
        if best_diff > angle_thresh:
            continue
        
        # Project onto dominant angle direction
        a = angles[best_idx]
        cos_a, sin_a = math.cos(a), math.sin(a)
        
        # Along-axis and perpendicular projections
        p1_along = x1 * cos_a + z1 * sin_a
        p1_perp = -x1 * sin_a + z1 * cos_a
        p2_along = x2 * cos_a + z2 * sin_a
        p2_perp = -x2 * sin_a + z2 * cos_a
        
        along_min = min(p1_along, p2_along)
        along_max = max(p1_along, p2_along)
        perp_avg = (p1_perp + p2_perp) / 2
        
        assigned[best_idx].append((along_min, along_max, perp_avg, length))
    
    # For each angle, cluster by perpendicular offset, then merge along-axis overlaps
    merged_lines = []
    
    for angle_idx, segs in assigned.items():
        if not segs:
            continue
        
        a = angles[angle_idx]
        cos_a, sin_a = math.cos(a), math.sin(a)
        
        # Sort by perp offset and cluster
        segs.sort(key=lambda s: s[2])
        
        clusters = []
        current_cluster = [segs[0]]
        
        for s in segs[1:]:
            if abs(s[2] - current_cluster[-1][2]) < dist_thresh:
                current_cluster.append(s)
            else:
                clusters.append(current_cluster)
                current_cluster = [s]
        clusters.append(current_cluster)
        
        # For each cluster, merge overlapping along-axis segments
        for cluster in clusters:
            perp = np.mean([s[2] for s in cluster])
            
            # Sort by along_min and merge overlapping
            intervals = sorted([(s[0], s[1]) for s in cluster])
            merged = [list(intervals[0])]
            for start, end in intervals[1:]:
                if start <= merged[-1][1] + 0.1:  # small gap tolerance
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            
            # Convert back to world coordinates, filter short
            for along_min, along_max in merged:
                length = along_max - along_min
                if length < 0.3:  # minimum wall segment length
                    continue
                
                x1 = along_min * cos_a - perp * sin_a
                z1 = along_min * sin_a + perp * cos_a
                x2 = along_max * cos_a - perp * sin_a
                z2 = along_max * sin_a + perp * cos_a
                
                merged_lines.append({
                    'x1': x1, 'z1': z1, 'x2': x2, 'z2': z2,
                    'length': length,
                    'angle_idx': angle_idx,
                    'angle': a,
                    'perp': perp
                })
    
    print(f"  Merged wall segments: {len(merged_lines)} (from {len(segments)} raw)")
    return merged_lines


def find_wall_intersections(walls, max_dist=0.3):
    """
    Find intersection points between walls of different angles.
    These become candidate room corners.
    """
    intersections = []
    
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            w1, w2 = walls[i], walls[j]
            
            # Skip near-parallel walls
            angle_diff = abs(w1['angle'] - w2['angle']) % math.pi
            if angle_diff < math.radians(15) or angle_diff > math.radians(165):
                continue
            
            # Line-line intersection
            dx1 = w1['x2'] - w1['x1']
            dz1 = w1['z2'] - w1['z1']
            dx2 = w2['x2'] - w2['x1']
            dz2 = w2['z2'] - w2['z1']
            
            det = dx1 * dz2 - dz1 * dx2
            if abs(det) < 1e-10:
                continue
            
            t = ((w2['x1'] - w1['x1']) * dz2 - (w2['z1'] - w1['z1']) * dx2) / det
            s = ((w2['x1'] - w1['x1']) * dz1 - (w2['z1'] - w1['z1']) * dx1) / det
            
            # Check if intersection is near both segments (with tolerance)
            tol = max_dist / max(w1['length'], 0.1)
            stol = max_dist / max(w2['length'], 0.1)
            
            if -tol <= t <= 1 + tol and -stol <= s <= 1 + stol:
                ix = w1['x1'] + t * dx1
                iz = w1['z1'] + t * dz1
                intersections.append({
                    'x': ix, 'z': iz,
                    'walls': (i, j),
                    't': t, 's': s
                })
    
    print(f"  Wall intersections: {len(intersections)}")
    return intersections


def build_planar_graph(walls, intersections):
    """
    Build a planar graph where:
    - Nodes = wall endpoints + intersection points
    - Edges = wall segments between consecutive nodes
    
    Returns adjacency list.
    """
    # Collect all nodes
    nodes = []
    node_map = {}  # (rounded_x, rounded_z) -> node_idx
    
    def add_node(x, z):
        key = (round(x, 3), round(z, 3))
        if key in node_map:
            return node_map[key]
        idx = len(nodes)
        nodes.append((x, z))
        node_map[key] = idx
        return idx
    
    # Add wall endpoints
    for w in walls:
        add_node(w['x1'], w['z1'])
        add_node(w['x2'], w['z2'])
    
    # Add intersections
    for inter in intersections:
        add_node(inter['x'], inter['z'])
    
    # Build edges: for each wall, split at intersection points and connect consecutive nodes
    adj = defaultdict(set)
    
    for wi, w in enumerate(walls):
        # Collect all points on this wall
        points_on_wall = [(0.0, add_node(w['x1'], w['z1']))]
        points_on_wall.append((1.0, add_node(w['x2'], w['z2'])))
        
        for inter in intersections:
            if wi in inter['walls']:
                t = inter['t'] if inter['walls'][0] == wi else inter['s']
                if -0.01 <= t <= 1.01:
                    nid = add_node(inter['x'], inter['z'])
                    points_on_wall.append((t, nid))
        
        points_on_wall.sort()
        
        # Connect consecutive points
        for k in range(len(points_on_wall) - 1):
            n1 = points_on_wall[k][1]
            n2 = points_on_wall[k + 1][1]
            if n1 != n2:
                adj[n1].add(n2)
                adj[n2].add(n1)
    
    print(f"  Graph: {len(nodes)} nodes, {sum(len(v) for v in adj.values()) // 2} edges")
    return nodes, adj


def find_minimal_cycles(nodes, adj, max_cycle_len=20):
    """
    Find minimal cycles (faces) in the planar graph.
    Uses the planar face traversal algorithm (left-turn rule).
    """
    if not nodes or not adj:
        return []
    
    # For each directed edge, find the "next" edge by turning left (counterclockwise)
    # Sort neighbors of each node by angle
    def angle_to(n1, n2):
        return math.atan2(nodes[n2][1] - nodes[n1][1], nodes[n2][0] - nodes[n1][0])
    
    sorted_neighbors = {}
    for n in adj:
        neighbors = sorted(adj[n], key=lambda nb: angle_to(n, nb))
        sorted_neighbors[n] = neighbors
    
    # For each directed edge (u, v), next edge turns left: 
    # find v's neighbor after u in clockwise order
    def next_edge(u, v):
        neighbors = sorted_neighbors.get(v, [])
        if not neighbors:
            return None
        # Find u in v's sorted neighbors
        try:
            idx = neighbors.index(u)
        except ValueError:
            return None
        # Next in clockwise = previous in counterclockwise sorted list
        next_idx = (idx - 1) % len(neighbors)
        return neighbors[next_idx]
    
    # Traverse all faces
    visited_edges = set()
    cycles = []
    
    for u in adj:
        for v in adj[u]:
            if (u, v) in visited_edges:
                continue
            
            # Trace cycle
            cycle = [u]
            current = u
            nxt = v
            
            for _ in range(max_cycle_len):
                if (current, nxt) in visited_edges:
                    break
                visited_edges.add((current, nxt))
                cycle.append(nxt)
                
                nn = next_edge(current, nxt)
                if nn is None:
                    break
                current = nxt
                nxt = nn
                
                if nxt == cycle[0]:
                    visited_edges.add((current, nxt))
                    # Found a cycle!
                    if len(cycle) >= 3:
                        cycles.append(cycle)
                    break
    
    print(f"  Raw cycles found: {len(cycles)}")
    return cycles


def cycle_to_polygon(cycle, nodes):
    """Convert node cycle to polygon coordinates."""
    return np.array([(nodes[n][0], nodes[n][1]) for n in cycle])


def polygon_area_signed(pts):
    n = len(pts)
    a = 0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return a / 2


def polygon_area(pts):
    return abs(polygon_area_signed(pts))


def polygon_centroid(pts):
    return pts.mean(axis=0)


def classify_room(polygon, area):
    xs, zs = polygon[:, 0], polygon[:, 1]
    w, h = xs.max() - xs.min(), zs.max() - zs.min()
    aspect = max(w, h) / (min(w, h) + 0.01)
    if area < 3:
        return "closet"
    if area < 5:
        return "hallway" if aspect > 2.0 else "bathroom"
    if aspect > 2.5:
        return "hallway"
    return "room"


def render_floorplan(rooms, walls, angles, output_path, title="v41"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    ax.set_facecolor('white')
    colors = ['#E8E8E8', '#F0F0F0', '#E0E0E0', '#F5F5F5', '#EBEBEB',
              '#E3E3E3', '#F2F2F2', '#EDEDED']
    
    # Draw walls as thick lines
    for w in walls:
        ax.plot([w['x1'], w['x2']], [w['z1'], w['z2']], 'k-', linewidth=1.0, alpha=0.3)
    
    # Draw rooms
    for i, room in enumerate(rooms):
        poly = room['polygon']
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:, 0], pc[:, 1], color=colors[i % len(colors)], alpha=0.5)
        ax.plot(pc[:, 0], pc[:, 1], 'k-', linewidth=2.5)
        cx, cz = poly.mean(axis=0)
        name = room.get('name', f'Room {i+1}')
        area = room.get('area_m2', 0)
        nv = room.get('vertices', 0)
        ax.text(cx, cz, f"{name}\n{area:.1f}m²\n({nv}v)", ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    angle_strs = [f"{math.degrees(a):.0f}°" for a in angles]
    ax.text(0.02, 0.98, f"Wall angles: {', '.join(angle_strs)}", transform=ax.transAxes,
            fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([xlim[0]+0.5, xlim[0]+1.5], [ylim[0]+0.3]*2, 'k-', linewidth=3)
    ax.text(xlim[0]+1.0, ylim[0]+0.15, '1m', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def render_debug(mesh, wall_mask_faces, segments, walls, nodes, adj, 
                 cycles, rooms, angles, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    verts = mesh.vertices
    
    # 1. All mesh edges projected to XZ
    ax = axes[0, 0]
    ax.set_title('Mesh XZ projection (sample)')
    sample_faces = mesh.faces[::10]  # subsample for speed
    for face in sample_faces:
        for k in range(3):
            i, j = face[k], face[(k+1)%3]
            ax.plot([verts[i,0], verts[j,0]], [verts[i,2], verts[j,2]], 
                    'b-', linewidth=0.1, alpha=0.1)
    ax.set_aspect('equal')
    
    # 2. Wall face edges
    ax = axes[0, 1]
    ax.set_title(f'Wall edges ({len(segments)} segs)')
    for seg in segments[::3]:  # subsample
        ax.plot([seg[0], seg[2]], [seg[1], seg[3]], 'r-', linewidth=0.2, alpha=0.3)
    ax.set_aspect('equal')
    
    # 3. Merged walls colored by angle
    ax = axes[0, 2]
    ax.set_title(f'Merged walls ({len(walls)})')
    angle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    for w in walls:
        c = angle_colors[w['angle_idx'] % len(angle_colors)]
        ax.plot([w['x1'], w['x2']], [w['z1'], w['z2']], '-', color=c, linewidth=1.5, alpha=0.7)
    ax.set_aspect('equal')
    
    # 4. Graph
    ax = axes[1, 0]
    ax.set_title(f'Planar graph ({len(nodes)}n, {sum(len(v) for v in adj.values())//2}e)')
    for n1 in adj:
        for n2 in adj[n1]:
            if n1 < n2:
                ax.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], 
                        'k-', linewidth=0.5)
    for x, z in nodes:
        ax.plot(x, z, 'r.', markersize=2)
    ax.set_aspect('equal')
    
    # 5. Cycles
    ax = axes[1, 1]
    ax.set_title(f'Cycles ({len(cycles)})')
    cycle_colors = plt.cm.Set3(np.linspace(0, 1, max(len(cycles), 1)))
    for ci, cycle in enumerate(cycles[:20]):
        poly = cycle_to_polygon(cycle, nodes)
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:,0], pc[:,1], color=cycle_colors[ci % len(cycle_colors)], alpha=0.3)
        ax.plot(pc[:,0], pc[:,1], '-', linewidth=0.8)
    ax.set_aspect('equal')
    
    # 6. Final rooms
    ax = axes[1, 2]
    ax.set_title(f'Rooms ({len(rooms)})')
    room_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                   '#DDA0DD', '#98D8C8', '#F7DC6F']
    for i, room in enumerate(rooms):
        poly = room['polygon']
        pc = np.vstack([poly, poly[0]])
        ax.fill(pc[:,0], pc[:,1], color=room_colors[i % len(room_colors)], alpha=0.4)
        ax.plot(pc[:,0], pc[:,1], 'k-', linewidth=2)
        cx, cz = poly.mean(axis=0)
        ax.text(cx, cz, f"{room.get('name','?')}\n{room.get('area_m2',0):.1f}m²",
                ha='center', va='center', fontsize=7)
    ax.set_aspect('equal')
    
    plt.suptitle('v41 Wall Segment Graph — Debug', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--normal-thresh', type=float, default=0.5,
                        help='Max abs(normal.y) for wall faces (default 0.5)')
    parser.add_argument('--min-wall-length', type=float, default=0.3,
                        help='Minimum merged wall segment length in meters')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output) if args.output else script_dir / 'results' / 'v41_wall_segment_graph'
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load_mesh(str(mesh_path))
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("\nStep 1: Identify wall faces...")
    wall_mask = extract_wall_faces(mesh, args.normal_thresh)

    print("\nStep 2: Project wall edges to XZ plane...")
    segments = project_wall_edges(mesh, wall_mask)

    print("\nStep 3: Find dominant wall angles...")
    angles = find_dominant_angles(segments)

    print("\nStep 4: Merge collinear segments...")
    walls = merge_collinear_segments(segments, angles, dist_thresh=0.12)
    
    # Filter short walls
    walls = [w for w in walls if w['length'] >= args.min_wall_length]
    print(f"  After length filter (>={args.min_wall_length}m): {len(walls)}")

    print("\nStep 5: Find wall intersections...")
    intersections = find_wall_intersections(walls, max_dist=0.2)

    print("\nStep 6: Build planar graph...")
    nodes, adj = build_planar_graph(walls, intersections)

    print("\nStep 7: Find minimal cycles (rooms)...")
    cycles = find_minimal_cycles(nodes, adj)
    
    # Filter cycles by area
    rooms = []
    for cycle in cycles:
        poly = cycle_to_polygon(cycle, nodes)
        area = polygon_area(poly)
        signed = polygon_area_signed(poly)
        
        # Only keep positive-area (counterclockwise) cycles with reasonable room size
        if 2.0 <= area <= 50.0 and signed > 0:
            rtype = classify_room(poly, area)
            rooms.append({
                'polygon': poly,
                'area_m2': round(area, 1),
                'type': rtype,
                'vertices': len(poly),
                'cycle': cycle
            })
    
    # Sort by area, keep non-overlapping
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # Remove heavily overlapping rooms (keep larger)
    filtered_rooms = []
    for room in rooms:
        poly = room['polygon']
        cx, cz = poly.mean(axis=0)
        
        overlaps = False
        for existing in filtered_rooms:
            epoly = existing['polygon']
            ecx, ecz = epoly.mean(axis=0)
            # Simple overlap check: centroid inside existing polygon
            from matplotlib.path import Path as MplPath
            if MplPath(epoly).contains_point((cx, cz)):
                overlaps = True
                break
        
        if not overlaps:
            filtered_rooms.append(room)
    
    rooms = filtered_rooms[:10]  # cap at 10
    print(f"  Rooms found: {len(rooms)}")
    
    # Name rooms
    rc, hc, bc, cc = 1, 1, 1, 1
    for room in rooms:
        t = room['type']
        if t == 'hallway':
            room['name'] = "Hallway" if hc == 1 else f"Hallway {hc}"
            hc += 1
        elif t == 'bathroom':
            room['name'] = "Bathroom" if bc == 1 else f"Bathroom {bc}"
            bc += 1
        elif t == 'closet':
            room['name'] = "Closet" if cc == 1 else f"Closet {cc}"
            cc += 1
        else:
            room['name'] = f"Room {rc}"
            rc += 1

    print("\nStep 8: Rendering...")
    mesh_name = mesh_path.stem
    
    render_floorplan(rooms, walls, angles,
                     out_dir / f"v41_{mesh_name}_plan.png",
                     f"v41 Wall Segment Graph — {mesh_name}")
    
    render_debug(mesh, wall_mask, segments, walls, nodes, adj, 
                 cycles, rooms, angles,
                 out_dir / f"v41_{mesh_name}_debug.png")

    # Copy main plan
    shutil.copy2(out_dir / f"v41_{mesh_name}_plan.png",
                 Path.home() / '.openclaw' / 'workspace' / 'latest_floorplan.png')

    # Save results
    total_area = sum(r['area_m2'] for r in rooms)
    results = {
        'approach': 'v41_wall_segment_graph',
        'dominant_angles_deg': [round(math.degrees(a), 1) for a in angles],
        'n_wall_segments': len(walls),
        'n_intersections': len(intersections),
        'n_graph_nodes': len(nodes),
        'n_cycles': len(cycles),
        'rooms': [{
            'name': r['name'],
            'area_m2': r['area_m2'],
            'type': r['type'],
            'vertices': r['vertices'],
            'polygon': r['polygon'].tolist()
        } for r in rooms],
        'total_area_m2': round(total_area, 1)
    }
    with open(out_dir / f"v41_{mesh_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== v41 Summary ===")
    print(f"  Dominant angles: {[f'{math.degrees(a):.0f}°' for a in angles]}")
    print(f"  Wall segments: {len(walls)}, Intersections: {len(intersections)}")
    print(f"  Graph: {len(nodes)} nodes, Cycles: {len(cycles)}")
    for r in rooms:
        print(f"  {r['name']}: {r['area_m2']}m², {r['vertices']}v ({r['type']})")
    print(f"  Total: {total_area:.1f}m²")


if __name__ == '__main__':
    main()
