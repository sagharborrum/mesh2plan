#!/usr/bin/env python3
"""v8: Polished floor plan with top-down orthographic mesh projection.
Uses vertex colors from textured mesh for background context."""

import json, sys, warnings
import numpy as np
import trimesh
from pathlib import Path
from collections import defaultdict
from scipy.spatial import ConvexHull
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

def collect_wall_points(mesh, n_slices=25):
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
    n_bins = max(50, int((coords.max() - coords.min()) / 0.02))
    hist, bin_edges = np.histogram(coords, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    threshold = max(np.median(hist) * 3, min_inliers)
    
    peaks = []
    in_peak = False; pw, ps = 0, 0
    for i in range(len(hist)):
        if hist[i] > threshold:
            if not in_peak: in_peak = True; pw = 0; ps = 0
            pw += hist[i]; ps += bin_centers[i] * hist[i]
        else:
            if in_peak: peaks.append((ps/pw, pw)); in_peak = False
    if in_peak: peaks.append((ps/pw, pw))
    return peaks

def fit_manhattan_walls(points_xz, min_inliers=15, dist_thresh=0.04):
    if len(points_xz) < 50: return [], 0, np.empty((0,2))
    
    angle_deg = find_dominant_angle(points_xz)
    angle_rad = np.radians(angle_deg)
    rotated = rotate_points(points_xz, -angle_rad)
    
    raw_walls = []
    for axis in [0, 1]:
        peaks = find_wall_positions(rotated[:, axis], min_inliers, dist_thresh)
        other_axis = 1 - axis
        for wall_pos, n_pts in peaks:
            mask = np.abs(rotated[:, axis] - wall_pos) < dist_thresh * 2
            wall_pts = rotated[mask]
            if len(wall_pts) < min_inliers: continue
            other = wall_pts[:, other_axis]
            seg_start = np.percentile(other, 2)
            seg_end = np.percentile(other, 98)
            if seg_end - seg_start < 0.3: continue
            raw_walls.append({"axis": "x" if axis == 0 else "z", "position": float(wall_pos),
                "start_rot": float(seg_start), "end_rot": float(seg_end), "n_points": int(mask.sum())})
    
    # Merge nearby parallel walls
    merged = []
    used = set()
    for i, w1 in enumerate(raw_walls):
        if i in used: continue
        group = [w1]
        for j, w2 in enumerate(raw_walls):
            if j <= i or j in used or w1["axis"] != w2["axis"]: continue
            if abs(w1["position"] - w2["position"]) < 0.15:
                group.append(w2); used.add(j)
        total_pts = sum(g["n_points"] for g in group)
        avg_pos = sum(g["position"] * g["n_points"] for g in group) / total_pts
        ext_start = min(min(g["start_rot"], g["end_rot"]) for g in group)
        ext_end = max(max(g["start_rot"], g["end_rot"]) for g in group)
        mw = dict(group[0]); mw["position"] = avg_pos; mw["start_rot"] = ext_start
        mw["end_rot"] = ext_end; mw["n_points"] = total_pts
        merged.append(mw); used.add(i)
    
    # Convert to original coordinates
    for w in merged:
        if w["axis"] == "x":
            p1_rot = np.array([w["position"], w["start_rot"]])
            p2_rot = np.array([w["position"], w["end_rot"]])
        else:
            p1_rot = np.array([w["start_rot"], w["position"]])
            p2_rot = np.array([w["end_rot"], w["position"]])
        p1 = rotate_points(p1_rot, angle_rad)
        p2 = rotate_points(p2_rot, angle_rad)
        w["start"] = p1.tolist(); w["end"] = p2.tolist()
        w["length"] = float(np.linalg.norm(p2 - p1))
    
    return merged, angle_deg, rotated

def connect_walls(walls, angle_deg):
    """Build room polygon from wall intersections."""
    if len(walls) < 3: return None
    
    angle_rad = np.radians(angle_deg)
    x_walls = [w for w in walls if w["axis"] == "x"]
    z_walls = [w for w in walls if w["axis"] == "z"]
    
    if not x_walls or not z_walls: return None
    
    # All perpendicular intersections
    intersections = []
    for xw in x_walls:
        for zw in z_walls:
            x, z = xw["position"], zw["position"]
            # Check if within reasonable range of both walls
            xw_range = sorted([xw["start_rot"], xw["end_rot"]])
            zw_range = sorted([zw["start_rot"], zw["end_rot"]])
            ext = 0.5
            if xw_range[0]-ext <= z <= xw_range[1]+ext and zw_range[0]-ext <= x <= zw_range[1]+ext:
                intersections.append([x, z])
    
    if len(intersections) < 3:
        # Fallback: all wall endpoints
        pts = []
        for w in walls: pts.extend([w["start"], w["end"]])
        pts = np.array(pts)
        try:
            hull = ConvexHull(pts)
            exterior = pts[hull.vertices].tolist()
            exterior.append(exterior[0])
            from shapely.geometry import Polygon
            poly = Polygon(exterior)
            return {"exterior": exterior, "area": float(poly.area), "perimeter": float(poly.length)}
        except: return None
    
    pts = np.array(intersections)
    # Rotate to aligned space for hull
    pts_rot = np.array([[p[0], p[1]] for p in pts])
    
    try:
        hull = ConvexHull(pts_rot)
        hull_pts_rot = pts_rot[hull.vertices]
        # Convert back to original coords
        hull_pts = rotate_points(hull_pts_rot, angle_rad)
        exterior = hull_pts.tolist()
        exterior.append(exterior[0])
        from shapely.geometry import Polygon
        poly = Polygon(exterior)
        if not poly.is_valid: poly = poly.buffer(0)
        return {"exterior": exterior, "area": float(poly.area), "perimeter": float(poly.length),
                "dimensions": [float(hull_pts[:,0].max()-hull_pts[:,0].min()), float(hull_pts[:,1].max()-hull_pts[:,1].min())]}
    except: return None

def detect_gaps(walls, rotated_pts, angle_deg, gap_threshold=0.3):
    angle_rad = np.radians(angle_deg)
    gaps = []
    for w in walls:
        if w["axis"] == "x":
            mask = np.abs(rotated_pts[:, 0] - w["position"]) < 0.06
            wall_pts = rotated_pts[mask, 1]
        else:
            mask = np.abs(rotated_pts[:, 1] - w["position"]) < 0.06
            wall_pts = rotated_pts[mask, 0]
        if len(wall_pts) < 10: continue
        wall_pts = np.sort(wall_pts)
        diffs = np.diff(wall_pts)
        for i, d in enumerate(diffs):
            if d > gap_threshold:
                gap_mid = (wall_pts[i] + wall_pts[i+1]) / 2
                if w["axis"] == "x":
                    p1 = rotate_points(np.array([w["position"], wall_pts[i]]), angle_rad)
                    p2 = rotate_points(np.array([w["position"], wall_pts[i+1]]), angle_rad)
                    mid = rotate_points(np.array([w["position"], gap_mid]), angle_rad)
                else:
                    p1 = rotate_points(np.array([wall_pts[i], w["position"]]), angle_rad)
                    p2 = rotate_points(np.array([wall_pts[i+1], w["position"]]), angle_rad)
                    mid = rotate_points(np.array([gap_mid, w["position"]]), angle_rad)
                gw = float(d)
                gtype = "door" if 0.6 < gw < 1.3 else ("window" if 0.3 < gw < 2.0 else "opening")
                gaps.append({"type": gtype, "width": gw, "start": p1.tolist(), "end": p2.tolist(), "mid": mid.tolist(), "wall_axis": w["axis"]})
    return gaps

def generate_topdown_raster(mesh, resolution=512):
    """Generate a top-down rasterized view of the mesh for floor plan background."""
    bounds = mesh.bounds
    x_range = bounds[1][0] - bounds[0][0]
    z_range = bounds[1][2] - bounds[0][2]
    
    # We'll sample the mesh from above and store depth/color
    aspect = z_range / x_range if x_range > 0 else 1
    w = resolution
    h = int(resolution * aspect)
    
    # Create rays from above
    x_coords = np.linspace(bounds[0][0], bounds[1][0], w)
    z_coords = np.linspace(bounds[0][2], bounds[1][2], h)
    
    # Grid of ray origins (shooting down along Y)
    xx, zz = np.meshgrid(x_coords, z_coords)
    origins = np.column_stack([xx.ravel(), np.full(w*h, bounds[1][UP_AXIS] + 1), zz.ravel()])
    directions = np.tile([0, -1, 0], (w*h, 1)).astype(float)
    
    # Ray cast
    print(f"  Raycasting {w}x{h} top-down view...", flush=True)
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
    except Exception as e:
        print(f"  Raycast failed: {e}", flush=True)
        return None
    
    if len(locations) == 0:
        print("  No ray hits", flush=True)
        return None
    
    # Depth map
    depth_img = np.zeros((h, w), dtype=float)
    y_min, y_max = bounds[0][UP_AXIS], bounds[1][UP_AXIS]
    
    for loc, ray_idx in zip(locations, index_ray):
        row = ray_idx // w
        col = ray_idx % w
        depth = (loc[UP_AXIS] - y_min) / (y_max - y_min)
        depth_img[row, col] = depth
    
    print(f"  Hit {len(locations)} of {w*h} rays ({100*len(locations)/(w*h):.1f}%)", flush=True)
    
    return {
        "width": w, "height": h,
        "bounds": {"x_min": float(bounds[0][0]), "x_max": float(bounds[1][0]),
                   "z_min": float(bounds[0][2]), "z_max": float(bounds[1][2])},
        "depth": depth_img.tolist(),
    }

def extract_planes(mesh, max_planes=15):
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    planes = []; remaining = np.arange(len(face_centers))
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
        pts = face_centers[fidx]; local = pts - centroid
        pu, pv = local @ u, local @ v
        try:
            hull = ConvexHull(np.column_stack([pu, pv]))
            boundary = [(centroid + pu[j]*u + pv[j]*v).tolist() for j in hull.vertices]
            boundary.append(boundary[0])
        except: boundary = []
        planes.append({"id": i, "type": ptype, "normal": normal.tolist(), "centroid": centroid.tolist(),
            "area": float(area), "boundary": boundary, "basis_u": u.tolist(), "basis_v": v.tolist()})
        mask = np.ones(len(remaining), dtype=bool); mask[inliers] = False
        remaining = remaining[mask]
    return planes

def generate_svg(walls, room, gaps, measurements, dominant_angle):
    """Generate clean SVG floor plan."""
    margin = 80; scale = 120
    all_pts = []
    for w in walls: all_pts.extend([w["start"], w["end"]])
    if room and room.get("exterior"): all_pts.extend(room["exterior"])
    pts = np.array(all_pts)
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    w = (mx[0] - mn[0]) * scale + 2 * margin
    h = (mx[1] - mn[1]) * scale + 2 * margin
    def tx(p): return (p[0] - mn[0]) * scale + margin, (p[1] - mn[1]) * scale + margin
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">',
             f'<rect width="{w}" height="{h}" fill="white"/>']
    
    if room and room.get("exterior"):
        pts_svg = " ".join(f"{tx(p)[0]:.1f},{tx(p)[1]:.1f}" for p in room["exterior"])
        lines.append(f'<polygon points="{pts_svg}" fill="#f5f8fc" stroke="none"/>')
    
    wt = 0.15 * scale
    for wall in walls:
        x1, y1 = tx(wall["start"]); x2, y2 = tx(wall["end"])
        dx, dy = x2-x1, y2-y1; length = (dx*dx + dy*dy)**0.5
        if length < 1: continue
        nx, ny = -dy/length * wt/2, dx/length * wt/2
        pts_svg = f"{x1+nx:.1f},{y1+ny:.1f} {x2+nx:.1f},{y2+ny:.1f} {x2-nx:.1f},{y2-ny:.1f} {x1-nx:.1f},{y1-ny:.1f}"
        lines.append(f'<polygon points="{pts_svg}" fill="#2a2a2a" stroke="#111" stroke-width="0.5"/>')
    
    for gap in gaps:
        x1, y1 = tx(gap["start"]); x2, y2 = tx(gap["end"])
        dx, dy = x2-x1, y2-y1; length = (dx*dx+dy*dy)**0.5
        if length < 2: continue
        nx, ny = -dy/length * wt*0.6, dx/length * wt*0.6
        if gap["type"] == "door":
            lines.append(f'<rect x="{min(x1,x2)-abs(nx)*1.1:.1f}" y="{min(y1,y2)-abs(ny)*1.1:.1f}" width="{abs(x2-x1)+abs(nx)*2.2:.1f}" height="{abs(y2-y1)+abs(ny)*2.2:.1f}" fill="white" stroke="none"/>')
            r = length
            angle = np.degrees(np.arctan2(dy, dx))
            lines.append(f'<path d="M {x1:.1f},{y1:.1f} A {r:.1f},{r:.1f} 0 0 0 {x1+r*np.cos(np.radians(angle-90)):.1f},{y1+r*np.sin(np.radians(angle-90)):.1f}" fill="none" stroke="#2196F3" stroke-width="1" stroke-dasharray="4,2"/>')
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x1+r*np.cos(np.radians(angle-90)):.1f}" y2="{y1+r*np.sin(np.radians(angle-90)):.1f}" stroke="#2196F3" stroke-width="1"/>')
        elif gap["type"] == "window":
            lines.append(f'<rect x="{min(x1,x2)-abs(nx)*0.6:.1f}" y="{min(y1,y2)-abs(ny)*0.6:.1f}" width="{abs(x2-x1)+abs(nx)*1.2:.1f}" height="{abs(y2-y1)+abs(ny)*1.2:.1f}" fill="white" stroke="none"/>')
            lines.append(f'<line x1="{x1+nx/2:.1f}" y1="{y1+ny/2:.1f}" x2="{x2+nx/2:.1f}" y2="{y2+ny/2:.1f}" stroke="#333" stroke-width="2"/>')
            lines.append(f'<line x1="{x1-nx/2:.1f}" y1="{y1-ny/2:.1f}" x2="{x2-nx/2:.1f}" y2="{y2-ny/2:.1f}" stroke="#333" stroke-width="2"/>')
    
    for m in measurements:
        x1, y1 = tx(m["start"]); x2, y2 = tx(m["end"])
        dx, dy = x2-x1, y2-y1; length = (dx*dx+dy*dy)**0.5
        if length < 40: continue
        nx, ny = -dy/length, dx/length; off = 30
        lines.append(f'<line x1="{x1+nx*off:.1f}" y1="{y1+ny*off:.1f}" x2="{x2+nx*off:.1f}" y2="{y2+ny*off:.1f}" stroke="#bbb" stroke-width="0.5" stroke-dasharray="3,3"/>')
        mx, my = (x1+x2)/2 + nx*off, (y1+y2)/2 + ny*off
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        lines.append(f'<text x="{mx:.1f}" y="{my-5:.1f}" text-anchor="middle" font-size="11" font-family="system-ui" fill="#666" transform="rotate({angle:.1f},{mx:.1f},{my-5:.1f})">{m["length"]:.2f}m</text>')
    
    if room:
        ext = [tx(p) for p in room["exterior"]]
        rcx = sum(p[0] for p in ext)/len(ext); rcy = sum(p[1] for p in ext)/len(ext)
        lines.append(f'<text x="{rcx:.1f}" y="{rcy:.1f}" text-anchor="middle" font-size="20" font-weight="bold" fill="#333">{room["area"]:.1f} m²</text>')
        lines.append(f'<text x="{rcx:.1f}" y="{rcy+20:.1f}" text-anchor="middle" font-size="13" fill="#999">({room["area"]*10.764:.0f} ft²)</text>')
    
    sb_x, sb_y, sb_w = margin, h-25, 1*scale
    lines.append(f'<line x1="{sb_x}" y1="{sb_y}" x2="{sb_x+sb_w}" y2="{sb_y}" stroke="#666" stroke-width="2"/>')
    lines.append(f'<text x="{sb_x+sb_w/2}" y="{sb_y+15}" text-anchor="middle" font-size="10" fill="#666">1m</text>')
    lines.append('</svg>')
    return "\n".join(lines)

def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "data/room_scan/2026_02_09_19_03_38/export_refined.obj"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/floorplan_v8.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, process=True)
    print(f"Mesh: {len(mesh.vertices)}v, {len(mesh.faces)}f", flush=True)
    
    print("\n── Planes ──", flush=True)
    planes = extract_planes(mesh)
    by_type = defaultdict(int)
    for p in planes: by_type[p["type"]] += 1
    print(f"  {len(planes)} planes: {dict(by_type)}", flush=True)
    
    print("\n── Cross-Sections ──", flush=True)
    all_xz, slice_data = collect_wall_points(mesh, n_slices=25)
    print(f"  Total: {len(all_xz)} XZ points", flush=True)
    
    print("\n── Manhattan Walls ──", flush=True)
    walls, dominant_angle, rotated_pts = fit_manhattan_walls(all_xz)
    for w in walls:
        print(f"    {w['axis']}: {w['length']:.2f}m ({w['n_points']} pts)", flush=True)
    
    print("\n── Room Polygon ──", flush=True)
    room = connect_walls(walls, dominant_angle)
    if room:
        print(f"  {room['area']:.2f}m², perimeter: {room['perimeter']:.2f}m", flush=True)
    
    print("\n── Gaps ──", flush=True)
    gaps = detect_gaps(walls, rotated_pts, dominant_angle)
    by_t = defaultdict(int)
    for g in gaps: by_t[g["type"]] += 1
    print(f"  {len(gaps)}: {dict(by_t)}", flush=True)
    
    # Top-down rasterization
    print("\n── Top-Down View ──", flush=True)
    topdown = generate_topdown_raster(mesh, resolution=400)
    
    measurements = [{"start": w["start"], "end": w["end"],
        "mid": [(w["start"][0]+w["end"][0])/2, (w["start"][1]+w["end"][1])/2],
        "length": w["length"]} for w in walls if w["length"] > 0.5]
    
    print("\n── SVG ──", flush=True)
    svg = generate_svg(walls, room, gaps, measurements, dominant_angle)
    svg_path = output_path.replace('.json', '.svg')
    with open(svg_path, 'w') as f: f.write(svg)
    print(f"  {svg_path}", flush=True)
    
    result = {
        "mesh": {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "total_area": float(mesh.area),
            "bounds_min": mesh.bounds[0].tolist(), "bounds_max": mesh.bounds[1].tolist()},
        "planes": planes, "slices": slice_data, "walls": walls,
        "room": room, "gaps": gaps, "measurements": measurements,
        "dominant_angle": float(dominant_angle),
        "topdown": topdown,
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"  {output_path}", flush=True)

if __name__ == "__main__":
    main()
