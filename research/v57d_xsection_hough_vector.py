#!/usr/bin/env python3
"""
mesh2plan v57d - Cross-Section Wall Detection + Hough Line Room Partition

Combines:
- v56b's cross-section + skeleton + gap bridging → clean wall image
- v52's Hough line → wall selection → Shapely polygonize → clean rooms

The wall image from cross-sections is much cleaner than density-based.
The Hough+Shapely approach gives perfectly straight walls.
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import cv2
from scipy import ndimage
from shapely.geometry import LineString, Polygon, MultiPoint
from shapely.ops import polygonize, unary_union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', '-o', default='results/v57d')
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.resolution
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min
    
    pad = 0.3
    x_min_w = verts[:, 0].min() - pad
    x_max_w = verts[:, 0].max() + pad
    z_min_w = verts[:, 2].min() - pad
    z_max_w = verts[:, 2].max() + pad
    W_img = int((x_max_w - x_min_w) / res) + 1
    H_img = int((z_max_w - z_min_w) / res) + 1
    print(f"  Image: {W_img}×{H_img}, res={res}m/px")
    
    def px_to_world(px, py):
        return x_min_w + px * res, z_min_w + py * res
    
    # ═══ Phase 1: Cross-section wall image (from v56b) ═══
    print("\n--- Phase 1: Multi-slice cross-section ---")
    slice_heights = np.linspace(y_min + 0.3*y_range, y_min + 0.8*y_range, 15)
    xsection = np.zeros((H_img, W_img), dtype=np.uint8)
    
    for y_h in slice_heights:
        try:
            section = mesh.section(plane_origin=[0, y_h, 0], plane_normal=[0, 1, 0])
            if section is None:
                continue
        except:
            continue
        for entity in section.entities:
            pts = section.vertices[entity.points]
            for i in range(len(pts) - 1):
                x0 = int((pts[i][0] - x_min_w) / res)
                z0 = int((pts[i][2] - z_min_w) / res)
                x1 = int((pts[i+1][0] - x_min_w) / res)
                z1 = int((pts[i+1][2] - z_min_w) / res)
                if (0 <= x0 < W_img and 0 <= z0 < H_img and
                    0 <= x1 < W_img and 0 <= z1 < H_img):
                    cv2.line(xsection, (x0, z0), (x1, z1), 255, thickness=1)
    
    print(f"  Cross-section: {(xsection > 0).mean()*100:.1f}% wall pixels")
    
    # Add normal-filtered wall density
    normals = mesh.face_normals
    wall_mask_3d = np.abs(normals[:, 1]) < 0.5
    wall_centroids = verts[faces[wall_mask_3d]].mean(axis=1)
    wall_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((wall_centroids[:, 0] - x_min_w) / res).astype(int)
    zi = ((wall_centroids[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(wall_density, (zi[valid], xi[valid]), 1)
    
    wd_thresh = np.percentile(wall_density[wall_density > 0], 85)
    wd_strong = (wall_density >= wd_thresh).astype(np.uint8) * 255
    combined = np.maximum(xsection, wd_strong)
    
    # ═══ Phase 2: Skeleton + gap bridging (from v56b) ═══
    print("\n--- Phase 2: Skeleton + bridging ---")
    
    # Detect dominant angles first
    lines_raw = cv2.HoughLines(combined, 1, np.pi/180, threshold=60)
    if lines_raw is not None:
        angles_deg = np.degrees(lines_raw[:, 0, 1]) % 180
        hist, bins = np.histogram(angles_deg, bins=180, range=(0, 180))
        hist_s = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist_s, height=max(hist_s)*0.15, distance=20)
        peak_angles = bins[peaks] + 0.5
        peak_h = hist_s[peaks]
        top2 = peak_angles[np.argsort(peak_h)[::-1][:2]]
        angle1, angle2 = float(top2[0]), float(top2[1]) if len(top2) > 1 else float(top2[0])+90
    else:
        angle1, angle2 = 28.0, 120.0
    
    if angle1 > angle2:
        angle1, angle2 = angle2, angle1
    print(f"  Dominant angles: {angle1:.0f}° and {angle2:.0f}°")
    
    # Skeleton
    wall_dilated = cv2.dilate(combined, np.ones((2, 2), np.uint8))
    try:
        skeleton = cv2.ximgproc.thinning(wall_dilated)
    except:
        from skimage.morphology import skeletonize
        skeleton = (skeletonize(wall_dilated > 0) * 255).astype(np.uint8)
    
    skel_binary = (skeleton > 0).astype(np.uint8)
    kernel3 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
    neighbor_count = cv2.filter2D(skel_binary, cv2.CV_32F, kernel3)
    endpoints = (skel_binary > 0) & (neighbor_count >= 0.5) & (neighbor_count <= 1.5)
    ep_ys, ep_xs = np.where(endpoints)
    print(f"  Skeleton: {skel_binary.sum()} px, {len(ep_xs)} endpoints")
    
    # Bridge nearby endpoints
    bridged = combined.copy()
    max_gap_px = int(0.8 / res)
    min_gap_px = int(0.1 / res)
    n_bridges = 0
    
    if len(ep_xs) > 0:
        ep_pts = np.column_stack([ep_xs, ep_ys])
        for i in range(len(ep_pts)):
            dists = np.linalg.norm(ep_pts - ep_pts[i], axis=1)
            nearby = np.where((dists > min_gap_px) & (dists < max_gap_px))[0]
            for j in nearby:
                if j <= i:
                    continue
                dx = ep_pts[j][0] - ep_pts[i][0]
                dy = ep_pts[j][1] - ep_pts[i][1]
                gap_angle = np.degrees(np.arctan2(dy, dx)) % 180
                for wall_angle in [angle1, angle2]:
                    diff = abs(gap_angle - wall_angle)
                    diff = min(diff, 180 - diff)
                    if diff < 20:
                        cv2.line(bridged, tuple(ep_pts[i]), tuple(ep_pts[j]), 255, thickness=1)
                        n_bridges += 1
                        break
    
    # Small directional closing
    def make_line_kernel(angle_deg, length=8):
        k = np.zeros((length*2+1, length*2+1), dtype=np.uint8)
        dx, dy = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
        for t in range(-length, length+1):
            x, y = int(length + t*dx), int(length + t*dy)
            if 0 <= x < length*2+1 and 0 <= y < length*2+1:
                k[y, x] = 1
        return k
    
    for a in [angle1, angle2]:
        k = make_line_kernel(a, length=8)
        bridged = np.maximum(bridged, cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, k))
    
    wall_final = cv2.dilate(bridged, np.ones((2, 2), np.uint8))
    print(f"  Bridged {n_bridges} gaps, final wall: {(wall_final > 0).mean()*100:.1f}%")
    
    # ═══ Phase 3: Hough lines on clean wall image ═══
    print("\n--- Phase 3: Hough wall lines ---")
    
    # Use Standard Hough on the final wall image
    hough_lines = cv2.HoughLines(wall_final, rho=1, theta=np.pi/180, threshold=50)
    if hough_lines is None:
        print("  ERROR: No Hough lines found")
        return
    
    hough_lines = hough_lines[:, 0, :]
    print(f"  Raw Hough: {len(hough_lines)} lines")
    
    # Filter to dominant angles (±12°)
    filtered = []
    for rho, theta in hough_lines:
        td = np.degrees(theta) % 180
        diff1 = min(abs(td - angle1), 180 - abs(td - angle1))
        diff2 = min(abs(td - angle2), 180 - abs(td - angle2))
        if diff1 < 12 or diff2 < 12:
            filtered.append((rho, theta))
    print(f"  Angle-filtered: {len(filtered)} lines")
    
    # Cluster by rho within each family
    def cluster_lines_by_rho(lines, angle_ref, min_gap=15):
        """Cluster lines near angle_ref by rho using histogram peak-finding."""
        fam = [(r, t) for r, t in lines 
               if min(abs(np.degrees(t)%180 - angle_ref), 180 - abs(np.degrees(t)%180 - angle_ref)) < 12]
        if not fam:
            return []
        
        rhos = np.array([r for r, t in fam])
        print(f"    Rho range: {rhos.min():.0f} to {rhos.max():.0f} ({len(fam)} lines)")
        
        # Use histogram to find rho peaks
        rho_min, rho_max = rhos.min() - 5, rhos.max() + 5
        n_bins = int(rho_max - rho_min) + 1
        hist, bin_edges = np.histogram(rhos, bins=n_bins, range=(rho_min, rho_max))
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        
        from scipy.signal import find_peaks
        # Find peaks with minimum distance = min_gap pixels
        peaks, props = find_peaks(hist_smooth, height=max(hist_smooth)*0.08, distance=min_gap)
        peak_rhos = bin_edges[peaks] + 0.5
        peak_heights = hist_smooth[peaks]
        
        print(f"    Found {len(peaks)} rho peaks: {[f'{r:.0f}' for r in peak_rhos]}")
        
        # For each peak, collect nearby lines and score
        results = []
        for peak_rho in peak_rhos:
            cluster = [(r, t) for r, t in fam if abs(r - peak_rho) < min_gap/2]
            if not cluster:
                continue
            avg_rho = np.mean([r for r, t in cluster])
            avg_theta = np.mean([t for r, t in cluster])
            
            # Score: wall density along line
            cos_t = np.cos(avg_theta)
            sin_t = np.sin(avg_theta)
            density_score = 0
            max_run = 0
            cur_run = 0
            count = 0
            for t_step in range(-max(H_img, W_img), max(H_img, W_img)):
                x = int(avg_rho * cos_t + t_step * (-sin_t))
                y = int(avg_rho * sin_t + t_step * cos_t)
                if 0 <= x < W_img and 0 <= y < H_img:
                    count += 1
                    if wall_final[y, x] > 0:
                        density_score += 1
                        cur_run += 1
                        max_run = max(max_run, cur_run)
                    else:
                        cur_run = 0
            
            score = density_score * np.sqrt(max_run + 1)
            results.append((avg_rho, avg_theta, score, density_score, max_run, len(cluster)))
        
        return results
    
    walls_fam1 = cluster_lines_by_rho(filtered, angle1, min_gap=12)
    walls_fam2 = cluster_lines_by_rho(filtered, angle2, min_gap=12)
    
    # Score-based selection: top walls by composite score
    # But also ensure minimum wall count per family
    def select_top_walls(walls, max_n=5, min_score_frac=0.3):
        if not walls:
            return []
        walls.sort(key=lambda x: x[2], reverse=True)
        max_score = walls[0][2]
        selected = []
        for w in walls[:max_n]:
            if w[2] >= max_score * min_score_frac or len(selected) < 2:
                selected.append(w)
        return selected
    
    sel1 = select_top_walls(walls_fam1, max_n=4, min_score_frac=0.4)
    sel2 = select_top_walls(walls_fam2, max_n=4, min_score_frac=0.4)
    
    print(f"  Family 1 ({angle1:.0f}°): {len(walls_fam1)} clusters → {len(sel1)} walls")
    for r, t, s, d, mr, n in sel1:
        print(f"    ρ={r:.0f}, score={s:.0f} (density={d}, run={mr}, votes={n})")
    print(f"  Family 2 ({angle2:.0f}°): {len(walls_fam2)} clusters → {len(sel2)} walls")
    for r, t, s, d, mr, n in sel2:
        print(f"    ρ={r:.0f}, score={s:.0f} (density={d}, run={mr}, votes={n})")
    
    # ═══ Phase 4: Create apartment boundary ═══
    print("\n--- Phase 4: Apartment boundary ---")
    all_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((verts[:, 0] - x_min_w) / res).astype(int)
    zi = ((verts[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(all_density, (zi[valid], xi[valid]), 1)
    
    apt_mask = (ndimage.gaussian_filter(all_density, sigma=5) > 0.5).astype(np.uint8)
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    apt_mask = ndimage.binary_fill_holes(apt_mask).astype(np.uint8)
    apt_mask = cv2.erode(apt_mask, np.ones((3, 3), np.uint8))
    
    # Convert mask to Shapely polygon
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    boundary_coords = []
    for pt in largest[:, 0]:
        wx, wz = px_to_world(pt[0], pt[1])
        boundary_coords.append((wx, wz))
    boundary = Polygon(boundary_coords)
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    print(f"  Boundary area: {boundary.area:.1f}m²")
    
    # ═══ Phase 5: Shapely room partition ═══
    print("\n--- Phase 5: Room partition ---")
    
    def hough_to_world_line(rho, theta, extent=500):
        """Convert Hough line from pixel to world LineString."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = rho * cos_t
        y0 = rho * sin_t
        dx = -sin_t
        dy = cos_t
        px1, py1 = x0 - extent*dx, y0 - extent*dy
        px2, py2 = x0 + extent*dx, y0 + extent*dy
        wx1, wz1 = px_to_world(px1, py1)
        wx2, wz2 = px_to_world(px2, py2)
        return LineString([(wx1, wz1), (wx2, wz2)])
    
    wall_lines = []
    wall_meta = []
    for r, t, s, d, mr, n in sel1 + sel2:
        line = hough_to_world_line(r, t)
        clipped = line.intersection(boundary)
        if not clipped.is_empty:
            wall_lines.append(clipped)
            wall_meta.append({'rho': float(r), 'angle': float(np.degrees(t)),
                            'score': float(s), 'density': int(d), 'run': int(mr)})
    
    print(f"  {len(wall_lines)} wall lines clipped to boundary")
    
    # Polygonize
    all_geoms = wall_lines + [boundary.boundary]
    merged = unary_union(all_geoms)
    polys = list(polygonize(merged))
    
    # Filter rooms
    rooms = []
    for poly in polys:
        if poly.area < 1.5:
            continue
        if boundary.contains(poly.centroid) or boundary.intersection(poly).area > 0.5 * poly.area:
            rooms.append(poly)
    
    rooms.sort(key=lambda r: r.area, reverse=True)
    total_area = sum(r.area for r in rooms)
    
    # Classify
    room_data = []
    for room in rooms:
        area = room.area
        perim = room.length
        compact = perim ** 2 / (4 * np.pi * area) if area > 0 else 0
        n_verts = len(room.exterior.coords) - 1
        
        if compact > 2.5 and area < 8:
            name = "Hallway"
        elif area > 8:
            name = "Room"
        elif area > 4:
            name = "Bathroom"
        else:
            name = "Closet"
        
        room_data.append({
            'name': name, 'area': float(area), 'vertices': n_verts,
            'centroid': [float(room.centroid.x), float(room.centroid.y)],
            'compactness': float(compact)
        })
    
    # Deduplicate names
    counts = {}
    for rd in room_data:
        n = rd['name']
        counts[n] = counts.get(n, 0) + 1
        if counts[n] > 1:
            rd['name'] = f"{n} {counts[n]}"
    
    print(f"\n  {len(rooms)} rooms, {total_area:.1f}m² total")
    for rd in room_data:
        print(f"    {rd['name']}: {rd['area']:.1f}m² ({rd['vertices']}v, compact={rd['compactness']:.1f})")
    
    # ═══ Phase 6: Check for oversized rooms → wall recovery (from v52) ═══
    max_room_area = 11.0  # Target: no room > 11m²
    recovered = False
    
    # Rejected walls (not selected)
    rejected1 = [w for w in walls_fam1 if w not in sel1]
    rejected2 = [w for w in walls_fam2 if w not in sel2]
    
    for attempt in range(3):  # Max 3 recovery attempts
        oversized = [(i, r) for i, r in enumerate(rooms) if r.area > max_room_area]
        if not oversized:
            break
        
        print(f"\n  Wall recovery attempt {attempt+1}: {len(oversized)} oversized rooms")
        
        best_wall = None
        best_score = 0
        best_idx = -1
        
        for idx, room in oversized:
            # Find best rejected wall that crosses this room
            for w_list in [rejected1, rejected2]:
                for r, t, s, d, mr, n in w_list:
                    line = hough_to_world_line(r, t)
                    intersection = line.intersection(room)
                    if not intersection.is_empty and intersection.length > 0.5:
                        if s > best_score:
                            best_score = s
                            best_wall = (r, t, s, d, mr, n)
                            best_idx = idx
        
        if best_wall is None:
            break
        
        r, t, s, d, mr, n = best_wall
        print(f"  Recovering wall: ρ={r:.0f}, score={s:.0f}")
        line = hough_to_world_line(r, t)
        clipped = line.intersection(boundary)
        if not clipped.is_empty:
            wall_lines.append(clipped)
            wall_meta.append({'rho': float(r), 'angle': float(np.degrees(t)),
                            'score': float(s), 'density': int(d), 'run': int(mr), 'recovered': True})
            
            # Remove from rejected
            for w_list in [rejected1, rejected2]:
                if best_wall in w_list:
                    w_list.remove(best_wall)
            
            # Re-partition
            all_geoms = wall_lines + [boundary.boundary]
            merged = unary_union(all_geoms)
            polys = list(polygonize(merged))
            rooms = [p for p in polys if p.area >= 1.5 and 
                     (boundary.contains(p.centroid) or boundary.intersection(p).area > 0.5 * p.area)]
            rooms.sort(key=lambda r: r.area, reverse=True)
            recovered = True
    
    if recovered:
        total_area = sum(r.area for r in rooms)
        room_data = []
        for room in rooms:
            area = room.area
            perim = room.length
            compact = perim ** 2 / (4 * np.pi * area) if area > 0 else 0
            n_verts = len(room.exterior.coords) - 1
            if compact > 2.5 and area < 8:
                name = "Hallway"
            elif area > 8:
                name = "Room"
            elif area > 4:
                name = "Bathroom"
            else:
                name = "Closet"
            room_data.append({
                'name': name, 'area': float(area), 'vertices': n_verts,
                'centroid': [float(room.centroid.x), float(room.centroid.y)],
                'compactness': float(compact)
            })
        
        counts = {}
        for rd in room_data:
            n = rd['name']
            counts[n] = counts.get(n, 0) + 1
            if counts[n] > 1:
                rd['name'] = f"{n} {counts[n]}"
        
        print(f"\n  After recovery: {len(rooms)} rooms, {total_area:.1f}m² total")
        for rd in room_data:
            print(f"    {rd['name']}: {rd['area']:.1f}m² ({rd['vertices']}v)")
    
    # ═══ Render ═══
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: wall detection pipeline
    axes[0,0].set_title('Cross-Section + Wall Density')
    axes[0,0].imshow(combined, cmap='gray_r', origin='lower')
    axes[0,0].set_aspect('equal')
    
    axes[0,1].set_title(f'After Bridging ({n_bridges} bridges)')
    axes[0,1].imshow(bridged, cmap='gray_r', origin='lower')
    axes[0,1].set_aspect('equal')
    
    axes[0,2].set_title(f'Final Walls ({(wall_final > 0).mean()*100:.1f}%)')
    axes[0,2].imshow(wall_final, cmap='gray_r', origin='lower')
    axes[0,2].set_aspect('equal')
    
    # Draw Hough lines on wall image
    ax = axes[1,0]
    ax.set_title(f'Hough Lines ({len(wall_lines)} walls)')
    ax.imshow(wall_final, cmap='gray', origin='lower', alpha=0.4)
    for wm in wall_meta:
        r = wm['rho']
        t = np.radians(wm['angle'])
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        x0 = r * cos_t
        y0 = r * sin_t
        dx = -sin_t * 500
        dy = cos_t * 500
        color = 'lime' if wm.get('recovered') else 'red'
        ax.plot([x0-dx, x0+dx], [y0-dy, y0+dy], color=color, lw=1.5, alpha=0.8)
    ax.set_xlim(0, W_img)
    ax.set_ylim(0, H_img)
    ax.set_aspect('equal')
    
    # Boundary + wall lines (world coords)
    ax = axes[1,1]
    ax.set_title('Boundary + Walls (world)')
    ax.set_aspect('equal')
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', lw=2)
    for i, line in enumerate(wall_lines):
        if hasattr(line, 'geoms'):
            for g in line.geoms:
                x, y = g.xy
                ax.plot(x, y, 'r-', lw=1.5)
        else:
            x, y = line.xy
            ax.plot(x, y, 'r-', lw=1.5)
    
    # Room polygons
    ax = axes[1,2]
    room_colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    ax.set_title(f'v57d — {len(rooms)} rooms, {total_area:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    ax.set_aspect('equal')
    
    bx, by = boundary.exterior.xy
    ax.plot(bx, by, 'k-', lw=2)
    
    for i, room in enumerate(rooms):
        x, y = room.exterior.xy
        ax.fill(x, y, color=room_colors[i % len(room_colors)], alpha=0.6)
        ax.plot(x, y, 'k-', lw=1.5)
        if i < len(room_data):
            cx, cy = room.centroid.x, room.centroid.y
            name = room_data[i]['name']
            area = room_data[i]['area']
            ax.text(cx, cy, f'{name}\n{area:.1f}m²', ha='center', va='center',
                   fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Scale bar
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.2, ylim[0]+0.2], 'k-', lw=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.05, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir / 'floorplan.png'}")
    
    # Save JSON
    result = {
        'version': 'v57d_xsection_hough_vector',
        'angles': [float(angle1), float(angle2)],
        'num_walls': len(wall_lines),
        'num_rooms': len(rooms),
        'total_area': float(total_area),
        'rooms': room_data,
        'walls': wall_meta
    }
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_dir / 'result.json'}")


if __name__ == '__main__':
    main()
