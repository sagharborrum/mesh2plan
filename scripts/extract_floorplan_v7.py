#!/usr/bin/env python3
"""v7: Connected Manhattan floor plan with door detection + SVG export.
Builds on v6: merges duplicate walls, connects into closed polygon, detects gaps."""

import json, sys, warnings
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon, MultiPoint, box
from shapely.ops import unary_union, polygonize, nearest_points
import pyransac3d as pyrsc

warnings.filterwarnings("ignore", category=DeprecationWarning)
UP_AXIS = 1

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def rotate_points(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    if pts.ndim == 1: return R @ pts
    return (R @ pts.T).T

def slice_mesh_3d(mesh, height):
    normal = np.zeros(3); normal[UP_AXIS] = 1.0
    origin = np.zeros(3); origin[UP_AXIS] = height
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None: return None
        return section.vertices
    except:
        return None

def collect_wall_points(mesh, n_slices=20):
    bounds = mesh.bounds
    y_min, y_max = bounds[0][UP_AXIS], bounds[1][UP_AXIS]
    y_range = y_max - y_min
    heights = np.linspace(y_min + y_range * 0.15, y_max - y_range * 0.15, n_slices)
    
    all_xz = []
    slice_data = []
    for h in heights:
        pts_3d = slice_mesh_3d(mesh, h)
        if pts_3d is None: continue
        xz = pts_3d[:, [0, 2]]
        all_xz.append(xz)
        slice_data.append({"height": float(h), "points_xz": xz.tolist(), "n_points": len(xz)})
        print(f"  y={h:.3f}: {len(xz)} pts", flush=True)
    
    return np.vstack(all_xz) if all_xz else np.empty((0, 2)), slice_data

def find_dominant_angle(points_xz, step=1):
    """Find dominant wall angle using histogram sharpness."""
    best_angle, best_score = 0, 0
    for deg in range(0, 180, step):
        rad = np.radians(deg)
        rot = rotate_points(points_xz, -rad)
        xh, _ = np.histogram(rot[:, 0], bins=100)
        zh, _ = np.histogram(rot[:, 1], bins=100)
        score = np.sum(xh ** 2) + np.sum(zh ** 2)
        if score > best_score:
            best_score = score; best_angle = deg
    return best_angle

def find_wall_positions(coords, min_inliers=15, dist_thresh=0.04):
    """Find wall positions along one axis via histogram peak detection."""
    n_bins = max(50, int((coords.max() - coords.min()) / 0.02))
    hist, bin_edges = np.histogram(coords, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    threshold = max(np.median(hist) * 3, min_inliers)
    
    peaks = []
    in_peak = False
    pw, ps = 0, 0
    for i in range(len(hist)):
        if hist[i] > threshold:
            if not in_peak: in_peak = True; pw = 0; ps = 0
            pw += hist[i]; ps += bin_centers[i] * hist[i]
        else:
            if in_peak: peaks.append((ps/pw, pw)); in_peak = False
    if in_peak: peaks.append((ps/pw, pw))
    return peaks

def merge_nearby_walls(walls, merge_dist=0.15):
    """Merge walls that are close and parallel."""
    if not walls: return walls
    
    merged = []
    used = set()
    
    # Sort by axis then position
    for i, w1 in enumerate(walls):
        if i in used: continue
        group = [w1]
        for j, w2 in enumerate(walls):
            if j <= i or j in used: continue
            if w1["axis"] != w2["axis"]: continue
            # Same axis — check if positions are close
            if abs(w1["position"] - w2["position"]) < merge_dist:
                group.append(w2)
                used.add(j)
        
        # Merge group: use weighted average position, union of extents
        if len(group) == 1:
            merged.append(w1)
        else:
            total_pts = sum(g["n_points"] for g in group)
            avg_pos = sum(g["position"] * g["n_points"] for g in group) / total_pts
            
            # Union of extents in the other axis
            all_starts = [min(g["start_rot"], g["end_rot"]) for g in group]
            all_ends = [max(g["start_rot"], g["end_rot"]) for g in group]
            ext_start = min(all_starts)
            ext_end = max(all_ends)
            
            mw = dict(group[0])  # Copy first wall's properties
            mw["position"] = avg_pos
            mw["start_rot"] = ext_start
            mw["end_rot"] = ext_end
            mw["n_points"] = total_pts
            merged.append(mw)
        used.add(i)
    
    return merged

def fit_manhattan_walls(points_xz, min_inliers=15, dist_thresh=0.04):
    """Find Manhattan-aligned wall segments."""
    if len(points_xz) < 50: return [], 0, []
    
    angle_deg = find_dominant_angle(points_xz)
    angle_rad = np.radians(angle_deg)
    rotated = rotate_points(points_xz, -angle_rad)
    
    print(f"  Dominant angle: {angle_deg}°", flush=True)
    
    raw_walls = []
    
    for axis in [0, 1]:
        coords = rotated[:, axis]
        other_axis = 1 - axis
        peaks = find_wall_positions(coords, min_inliers, dist_thresh)
        
        for wall_pos, n_pts in peaks:
            mask = np.abs(rotated[:, axis] - wall_pos) < dist_thresh * 2
            wall_pts = rotated[mask]
            if len(wall_pts) < min_inliers: continue
            
            other = wall_pts[:, other_axis]
            seg_start = np.percentile(other, 2)
            seg_end = np.percentile(other, 98)
            if seg_end - seg_start < 0.3: continue
            
            raw_walls.append({
                "axis": "x" if axis == 0 else "z",
                "position": float(wall_pos),
                "start_rot": float(seg_start),
                "end_rot": float(seg_end),
                "n_points": int(mask.sum()),
            })
    
    # Merge nearby parallel walls
    walls = merge_nearby_walls(raw_walls, merge_dist=0.15)
    print(f"  {len(raw_walls)} raw walls → {len(walls)} after merge", flush=True)
    
    # Convert back to original coordinates
    for w in walls:
        if w["axis"] == "x":
            p1_rot = np.array([w["position"], w["start_rot"]])
            p2_rot = np.array([w["position"], w["end_rot"]])
        else:
            p1_rot = np.array([w["start_rot"], w["position"]])
            p2_rot = np.array([w["end_rot"], w["position"]])
        
        p1 = rotate_points(p1_rot, angle_rad)
        p2 = rotate_points(p2_rot, angle_rad)
        w["start"] = p1.tolist()
        w["end"] = p2.tolist()
        w["length"] = float(np.linalg.norm(p2 - p1))
    
    return walls, angle_deg, rotated

def connect_walls_into_polygon(walls, angle_deg):
    """Connect wall segments into a closed room polygon by extending to intersections."""
    if len(walls) < 3: return None
    
    angle_rad = np.radians(angle_deg)
    
    # Work in rotated (axis-aligned) space
    x_walls = sorted([w for w in walls if w["axis"] == "x"], key=lambda w: w["position"])
    z_walls = sorted([w for w in walls if w["axis"] == "z"], key=lambda w: w["position"])
    
    if not x_walls or not z_walls:
        return None
    
    # Find the outermost walls (room boundary)
    # For x-walls: leftmost and rightmost
    # For z-walls: bottom and top
    
    # Score walls by n_points * length (more prominent = more likely room boundary)
    def wall_score(w): return w["n_points"] * w["length"]
    
    # Get boundary walls
    x_walls_scored = sorted(x_walls, key=wall_score, reverse=True)
    z_walls_scored = sorted(z_walls, key=wall_score, reverse=True)
    
    # Take up to 4-6 walls that form the room boundary
    # Strategy: take the most prominent walls and try to form a closed polygon
    
    # Simple approach: use the 2 outermost x-walls and 2 outermost z-walls
    # to form a rectangular room, then add internal walls for L-shapes
    
    if len(x_walls) >= 2 and len(z_walls) >= 2:
        # Outermost walls
        x_left = min(x_walls, key=lambda w: w["position"])
        x_right = max(x_walls, key=lambda w: w["position"])
        z_bottom = min(z_walls, key=lambda w: w["position"])
        z_top = max(z_walls, key=lambda w: w["position"])
        
        # Corners from intersections
        corners_rot = [
            (x_left["position"], z_bottom["position"]),   # bottom-left
            (x_right["position"], z_bottom["position"]),   # bottom-right
            (x_right["position"], z_top["position"]),      # top-right
            (x_left["position"], z_top["position"]),       # top-left
        ]
        
        # Check for L-shape: are there intermediate walls?
        inner_x = [w for w in x_walls if w != x_left and w != x_right]
        inner_z = [w for w in z_walls if w != z_bottom and w != z_top]
        
        # Build polygon — start with rectangle, then carve out for L-shapes
        polygon_rot = list(corners_rot)
        
        # If there are inner walls, try to build L-shape
        if inner_x or inner_z:
            # More complex polygon — find the actual room shape
            # Use all wall segments as edges and try to build a closed polygon
            all_segments_rot = []
            for w in walls:
                if w["axis"] == "x":
                    all_segments_rot.append(((w["position"], w["start_rot"]), (w["position"], w["end_rot"])))
                else:
                    all_segments_rot.append(((w["start_rot"], w["position"]), (w["end_rot"], w["position"])))
            
            # Extend segments to intersect
            extended = extend_and_intersect(walls, x_left["position"], x_right["position"],
                                           z_bottom["position"], z_top["position"])
            if extended:
                polygon_rot = extended
        
        # Close polygon
        if polygon_rot[-1] != polygon_rot[0]:
            polygon_rot.append(polygon_rot[0])
        
        # Convert to original coordinates
        pts_rot = np.array(polygon_rot)
        pts_orig = rotate_points(pts_rot, angle_rad)
        exterior = pts_orig.tolist()
        
        # Compute area
        try:
            poly = Polygon(exterior)
            if not poly.is_valid:
                poly = poly.buffer(0)
            area = poly.area
            perimeter = poly.length
        except:
            area = 0; perimeter = 0
        
        return {
            "exterior": exterior,
            "area": float(area),
            "perimeter": float(perimeter),
            "corners_rot": polygon_rot,
        }
    
    return None

def extend_and_intersect(walls, x_min, x_max, z_min, z_max):
    """Try to build a polygon by extending wall segments to find intersections.
    Walk around the room boundary clockwise."""
    
    # All walls as infinite lines in rotated space
    segments = []
    for w in walls:
        if w["axis"] == "x":
            # Vertical line at x=pos, from z_start to z_end
            segments.append({
                "axis": "x", "pos": w["position"],
                "start": w["start_rot"], "end": w["end_rot"],
                "score": w["n_points"] * w["length"],
            })
        else:
            segments.append({
                "axis": "z", "pos": w["position"],
                "start": w["start_rot"], "end": w["end_rot"],
                "score": w["n_points"] * w["length"],
            })
    
    # Find all intersections between perpendicular walls
    intersections = []
    for i, s1 in enumerate(segments):
        for j, s2 in enumerate(segments):
            if i >= j: continue
            if s1["axis"] == s2["axis"]: continue
            
            # Perpendicular walls always intersect
            if s1["axis"] == "x":
                x, z = s1["pos"], s2["pos"]
            else:
                x, z = s2["pos"], s1["pos"]
            
            # Check if intersection is reasonably close to both segments
            # (within some extension tolerance)
            ext = 0.5  # meters extension tolerance
            ok1 = True  # Always accept for now
            ok2 = True
            
            intersections.append((x, z, i, j))
    
    if len(intersections) < 3:
        return None
    
    # Build convex hull of intersections to get room outline
    pts = np.array([(x, z) for x, z, _, _ in intersections])
    try:
        hull = ConvexHull(pts)
        polygon = [tuple(pts[v]) for v in hull.vertices]
        return polygon
    except:
        return None

def detect_gaps(walls, rotated_pts, angle_deg, gap_threshold=0.3):
    """Detect gaps in walls (potential doors/windows)."""
    angle_rad = np.radians(angle_deg)
    gaps = []
    
    for w in walls:
        if w["axis"] == "x":
            # Points near this wall
            mask = np.abs(rotated_pts[:, 0] - w["position"]) < 0.06
            wall_pts = rotated_pts[mask, 1]  # Z coords along wall
        else:
            mask = np.abs(rotated_pts[:, 1] - w["position"]) < 0.06
            wall_pts = rotated_pts[mask, 0]  # X coords along wall
        
        if len(wall_pts) < 10: continue
        
        # Sort and find gaps
        wall_pts = np.sort(wall_pts)
        diffs = np.diff(wall_pts)
        
        for i, d in enumerate(diffs):
            if d > gap_threshold:
                # Gap detected
                gap_start_1d = wall_pts[i]
                gap_end_1d = wall_pts[i + 1]
                gap_mid_1d = (gap_start_1d + gap_end_1d) / 2
                gap_width = float(d)
                
                # Convert to original coordinates
                if w["axis"] == "x":
                    p1_rot = np.array([w["position"], gap_start_1d])
                    p2_rot = np.array([w["position"], gap_end_1d])
                    mid_rot = np.array([w["position"], gap_mid_1d])
                else:
                    p1_rot = np.array([gap_start_1d, w["position"]])
                    p2_rot = np.array([gap_end_1d, w["position"]])
                    mid_rot = np.array([gap_mid_1d, w["position"]])
                
                p1 = rotate_points(p1_rot, angle_rad)
                p2 = rotate_points(p2_rot, angle_rad)
                mid = rotate_points(mid_rot, angle_rad)
                
                # Classify: door (0.6-1.2m), window (0.5-2.0m), other
                gap_type = "door" if 0.6 < gap_width < 1.3 else ("window" if 0.3 < gap_width < 2.0 else "opening")
                
                gaps.append({
                    "type": gap_type,
                    "width": gap_width,
                    "start": p1.tolist(),
                    "end": p2.tolist(),
                    "mid": mid.tolist(),
                    "wall_axis": w["axis"],
                })
    
    return gaps

def extract_planes(mesh, max_planes=15):
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    planes = []
    remaining = np.arange(len(face_centers))
    fitter = pyrsc.Plane()
    
    for i in range(max_planes):
        if len(remaining) < 200: break
        eq, inliers = fitter.fit(face_centers[remaining], thresh=0.015, maxIteration=1500)
        if len(inliers) < 200: break
        
        normal = np.array(eq[:3]); normal /= np.linalg.norm(normal)
        fidx = remaining[inliers]
        centroid = np.average(face_centers[fidx], axis=0, weights=face_areas[fidx])
        area = face_areas[fidx].sum()
        
        up = np.zeros(3); up[UP_AXIS] = 1.0
        dot = abs(np.dot(normal, up))
        ptype = "floor" if dot > 0.7 and normal[UP_AXIS] > 0 else ("ceiling" if dot > 0.7 else ("wall" if dot < 0.3 else "unknown"))
        
        if abs(normal[UP_AXIS]) > 0.9: u = np.array([1,0,0], dtype=float)
        else: u = np.cross(normal, np.array([0,1,0], dtype=float))
        u /= np.linalg.norm(u); v = np.cross(normal, u); v /= np.linalg.norm(v)
        
        pts = face_centers[fidx]
        local = pts - centroid
        pu, pv = local @ u, local @ v
        try:
            hull = ConvexHull(np.column_stack([pu, pv]))
            boundary = [(centroid + pu[j]*u + pv[j]*v).tolist() for j in hull.vertices]
            boundary.append(boundary[0])
        except: boundary = []
        
        planes.append({
            "id": i, "type": ptype, "normal": normal.tolist(),
            "centroid": centroid.tolist(), "area": float(area),
            "boundary": boundary, "basis_u": u.tolist(), "basis_v": v.tolist(),
        })
        
        mask = np.ones(len(remaining), dtype=bool); mask[inliers] = False
        remaining = remaining[mask]
    
    return planes

def generate_svg(walls, room, gaps, measurements, dominant_angle, bounds):
    """Generate SVG floor plan."""
    margin = 80
    scale = 100  # pixels per meter
    
    all_pts = []
    for w in walls: all_pts.extend([w["start"], w["end"]])
    if room and room.get("exterior"): all_pts.extend(room["exterior"])
    pts = np.array(all_pts)
    
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    w = (mx[0] - mn[0]) * scale + 2 * margin
    h = (mx[1] - mn[1]) * scale + 2 * margin
    
    def tx(p): return (p[0] - mn[0]) * scale + margin, (p[1] - mn[1]) * scale + margin
    
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">')
    lines.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    
    # Room fill
    if room and room.get("exterior"):
        pts_svg = " ".join(f"{tx(p)[0]:.1f},{tx(p)[1]:.1f}" for p in room["exterior"])
        lines.append(f'<polygon points="{pts_svg}" fill="#f0f7ff" stroke="none"/>')
    
    # Walls (thick lines)
    wall_thick = 0.12 * scale
    for wall in walls:
        x1, y1 = tx(wall["start"])
        x2, y2 = tx(wall["end"])
        dx, dy = x2-x1, y2-y1
        length = (dx*dx + dy*dy) ** 0.5
        if length < 1: continue
        nx, ny = -dy/length * wall_thick/2, dx/length * wall_thick/2
        
        pts_svg = f"{x1+nx:.1f},{y1+ny:.1f} {x2+nx:.1f},{y2+ny:.1f} {x2-nx:.1f},{y2-ny:.1f} {x1-nx:.1f},{y1-ny:.1f}"
        lines.append(f'<polygon points="{pts_svg}" fill="#333" stroke="#111" stroke-width="1"/>')
    
    # Gaps (doors/windows)
    for gap in gaps:
        x1, y1 = tx(gap["start"])
        x2, y2 = tx(gap["end"])
        if gap["type"] == "door":
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#2196F3" stroke-width="3" stroke-dasharray="6,3"/>')
            # Door arc
            mx, my = (x1+x2)/2, (y1+y2)/2
            r = ((x2-x1)**2 + (y2-y1)**2)**0.5 / 2
            lines.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="{r:.1f}" fill="none" stroke="#2196F3" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/>')
        elif gap["type"] == "window":
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#4CAF50" stroke-width="4"/>')
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="white" stroke-width="2"/>')
    
    # Measurements
    for m in measurements:
        x1, y1 = tx(m["start"])
        x2, y2 = tx(m["end"])
        dx, dy = x2-x1, y2-y1
        length = (dx*dx + dy*dy) ** 0.5
        if length < 30: continue
        nx, ny = -dy/length, dx/length
        off = 25
        
        lines.append(f'<line x1="{x1+nx*off:.1f}" y1="{y1+ny*off:.1f}" x2="{x2+nx*off:.1f}" y2="{y2+ny*off:.1f}" stroke="#999" stroke-width="0.5" stroke-dasharray="3,3"/>')
        
        mx, my = (x1+x2)/2 + nx*off, (y1+y2)/2 + ny*off
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        lines.append(f'<text x="{mx:.1f}" y="{my-4:.1f}" text-anchor="middle" font-size="11" font-family="system-ui" fill="#666" transform="rotate({angle:.1f},{mx:.1f},{my-4:.1f})">{m["length"]:.2f}m</text>')
    
    # Room area label
    if room and room.get("exterior"):
        ext = np.array(room["exterior"])
        cx = np.mean([tx(p)[0] for p in room["exterior"]])
        cy = np.mean([tx(p)[1] for p in room["exterior"]])
        lines.append(f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" font-size="18" font-weight="bold" font-family="system-ui" fill="#333">{room["area"]:.1f} m²</text>')
        lines.append(f'<text x="{cx:.1f}" y="{cy+18:.1f}" text-anchor="middle" font-size="12" font-family="system-ui" fill="#999">({room["area"]*10.764:.0f} ft²)</text>')
    
    # Scale bar
    sb_x, sb_y = margin, h - 20
    sb_w = 1 * scale
    lines.append(f'<line x1="{sb_x}" y1="{sb_y}" x2="{sb_x+sb_w}" y2="{sb_y}" stroke="#666" stroke-width="2"/>')
    lines.append(f'<line x1="{sb_x}" y1="{sb_y-5}" x2="{sb_x}" y2="{sb_y+5}" stroke="#666" stroke-width="1"/>')
    lines.append(f'<line x1="{sb_x+sb_w}" y1="{sb_y-5}" x2="{sb_x+sb_w}" y2="{sb_y+5}" stroke="#666" stroke-width="1"/>')
    lines.append(f'<text x="{sb_x+sb_w/2}" y="{sb_y+15}" text-anchor="middle" font-size="10" fill="#666">1m</text>')
    
    lines.append('</svg>')
    return "\n".join(lines)

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/room_scan/2026_02_09_19_03_38/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/floorplan_v7.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, process=True)
    print(f"Mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f, area={mesh.area:.1f}m²", flush=True)
    
    # 1. Planes
    print("\n── Planes ──", flush=True)
    planes = extract_planes(mesh)
    by_type = defaultdict(int)
    for p in planes: by_type[p["type"]] += 1
    print(f"  {len(planes)} planes: {dict(by_type)}", flush=True)
    
    # 2. Cross-section points
    print("\n── Cross-Sections ──", flush=True)
    all_xz, slice_data = collect_wall_points(mesh, n_slices=20)
    print(f"  Total: {len(all_xz)} XZ points", flush=True)
    
    # 3. Manhattan walls
    print("\n── Manhattan Walls ──", flush=True)
    walls, dominant_angle, rotated_pts = fit_manhattan_walls(all_xz)
    for w in walls:
        print(f"    {w['axis']}-wall: {w['length']:.2f}m ({w['n_points']} pts) @ pos={w['position']:.3f}", flush=True)
    
    # 4. Room polygon from connected walls
    print("\n── Room Polygon ──", flush=True)
    room = connect_walls_into_polygon(walls, dominant_angle)
    if room:
        print(f"  Area: {room['area']:.2f}m², perimeter: {room['perimeter']:.2f}m", flush=True)
        ext = np.array(room["exterior"])
        room["dimensions"] = [float(ext[:,0].max() - ext[:,0].min()), float(ext[:,1].max() - ext[:,1].min())]
    
    # 5. Gap detection
    print("\n── Gap Detection ──", flush=True)
    gaps = detect_gaps(walls, rotated_pts, dominant_angle, gap_threshold=0.3)
    by_type_gaps = defaultdict(int)
    for g in gaps: by_type_gaps[g["type"]] += 1
    print(f"  {len(gaps)} gaps: {dict(by_type_gaps)}", flush=True)
    for g in gaps:
        print(f"    {g['type']}: {g['width']:.2f}m on {g['wall_axis']}-wall", flush=True)
    
    # 6. Measurements
    measurements = []
    for w in walls:
        if w["length"] > 0.5:
            mid = [(w["start"][0]+w["end"][0])/2, (w["start"][1]+w["end"][1])/2]
            measurements.append({"start": w["start"], "end": w["end"], "mid": mid, "length": w["length"]})
    
    # 7. SVG export
    print("\n── SVG Export ──", flush=True)
    svg = generate_svg(walls, room, gaps, measurements, dominant_angle, mesh.bounds)
    svg_path = output_path.replace('.json', '.svg')
    with open(svg_path, 'w') as f: f.write(svg)
    print(f"  Saved: {svg_path}", flush=True)
    
    # 8. Save JSON
    result = {
        "mesh": {
            "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "total_area": float(mesh.area),
            "bounds_min": mesh.bounds[0].tolist(), "bounds_max": mesh.bounds[1].tolist(),
        },
        "planes": planes,
        "slices": slice_data,
        "walls": walls,
        "room": room,
        "gaps": gaps,
        "measurements": measurements,
        "dominant_angle": float(dominant_angle),
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"  Saved: {output_path}", flush=True)

if __name__ == "__main__":
    main()
