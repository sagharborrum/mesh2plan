#!/usr/bin/env python3
"""
mesh2plan v57e - v56b Wall Image + v52-style Hough Wall Selection + Raster Partition

Key insight: v56b produces the BEST wall image (cross-section + skeleton + bridging).
v52 has the BEST wall selection (density-scored, wall recovery for oversized rooms).
Use v56b's wall image as input to v52's Hough selection → draw selected wall lines 
on apartment mask → flood fill → rooms.

No Shapely needed. Pure raster. Clean rooms because walls are straight Hough lines.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', '-o', default='results/v57e')
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
    W = int((x_max_w - x_min_w) / res) + 1
    H = int((z_max_w - z_min_w) / res) + 1
    print(f"  Image: {W}×{H}")
    
    # ═══ Phase 1: Cross-section wall image ═══
    print("\n--- Phase 1: Cross-section ---")
    slice_heights = np.linspace(y_min + 0.3*y_range, y_min + 0.8*y_range, 15)
    xsection = np.zeros((H, W), dtype=np.uint8)
    for y_h in slice_heights:
        try:
            section = mesh.section(plane_origin=[0, y_h, 0], plane_normal=[0, 1, 0])
            if section is None: continue
        except: continue
        for entity in section.entities:
            pts = section.vertices[entity.points]
            for i in range(len(pts) - 1):
                x0 = int((pts[i][0] - x_min_w) / res)
                z0 = int((pts[i][2] - z_min_w) / res)
                x1 = int((pts[i+1][0] - x_min_w) / res)
                z1 = int((pts[i+1][2] - z_min_w) / res)
                if 0 <= x0 < W and 0 <= z0 < H and 0 <= x1 < W and 0 <= z1 < H:
                    cv2.line(xsection, (x0, z0), (x1, z1), 255, 1)
    
    # Normal-filtered wall density
    normals = mesh.face_normals
    wall_faces = np.abs(normals[:, 1]) < 0.5
    wc = verts[faces[wall_faces]].mean(axis=1)
    wall_density = np.zeros((H, W), dtype=np.float32)
    xi = ((wc[:, 0] - x_min_w) / res).astype(int)
    zi = ((wc[:, 2] - z_min_w) / res).astype(int)
    v = (xi >= 0) & (xi < W) & (zi >= 0) & (zi < H)
    np.add.at(wall_density, (zi[v], xi[v]), 1)
    
    wd_thresh = np.percentile(wall_density[wall_density > 0], 85)
    wd_strong = (wall_density >= wd_thresh).astype(np.uint8) * 255
    wall_img = np.maximum(xsection, wd_strong)
    
    # Skeleton + bridging
    hlines = cv2.HoughLines(wall_img, 1, np.pi/180, 60)
    if hlines is not None:
        ang = np.degrees(hlines[:, 0, 1]) % 180
        hist, bins = np.histogram(ang, 180, (0, 180))
        hs = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        from scipy.signal import find_peaks
        pks, _ = find_peaks(hs, height=max(hs)*0.15, distance=20)
        pk_a = bins[pks] + 0.5
        pk_h = hs[pks]
        top2 = pk_a[np.argsort(pk_h)[::-1][:2]]
        angle1, angle2 = sorted([float(top2[0]), float(top2[1]) if len(top2) > 1 else float(top2[0])+90])
    else:
        angle1, angle2 = 26.0, 118.0
    print(f"  Angles: {angle1:.0f}° and {angle2:.0f}°")
    
    # Skeleton
    dilated = cv2.dilate(wall_img, np.ones((2,2), np.uint8))
    try:
        skeleton = cv2.ximgproc.thinning(dilated)
    except:
        from skimage.morphology import skeletonize
        skeleton = (skeletonize(dilated > 0) * 255).astype(np.uint8)
    
    sb = (skeleton > 0).astype(np.uint8)
    k3 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
    nc = cv2.filter2D(sb, cv2.CV_32F, k3)
    eps = (sb > 0) & (nc >= 0.5) & (nc <= 1.5)
    ey, ex = np.where(eps)
    
    # Bridge endpoints
    bridged = wall_img.copy()
    n_bridges = 0
    max_gap = int(0.8 / res)
    min_gap = int(0.1 / res)
    if len(ex) > 0:
        ep = np.column_stack([ex, ey])
        for i in range(len(ep)):
            d = np.linalg.norm(ep - ep[i], axis=1)
            near = np.where((d > min_gap) & (d < max_gap))[0]
            for j in near:
                if j <= i: continue
                ga = np.degrees(np.arctan2(ep[j][1]-ep[i][1], ep[j][0]-ep[i][0])) % 180
                for wa in [angle1, angle2]:
                    diff = min(abs(ga - wa), 180 - abs(ga - wa))
                    if diff < 20:
                        cv2.line(bridged, tuple(ep[i]), tuple(ep[j]), 255, 1)
                        n_bridges += 1
                        break
    
    # Directional closing
    def line_kernel(a, l=8):
        k = np.zeros((l*2+1, l*2+1), np.uint8)
        dx, dy = np.cos(np.radians(a)), np.sin(np.radians(a))
        for t in range(-l, l+1):
            x, y = int(l + t*dx), int(l + t*dy)
            if 0 <= x < l*2+1 and 0 <= y < l*2+1: k[y, x] = 1
        return k
    for a in [angle1, angle2]:
        bridged = np.maximum(bridged, cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, line_kernel(a)))
    
    wall_final = cv2.dilate(bridged, np.ones((2,2), np.uint8))
    print(f"  Wall: {(wall_final > 0).mean()*100:.1f}%, {n_bridges} bridges")
    
    # ═══ Phase 2: Apartment mask ═══
    print("\n--- Phase 2: Apartment mask ---")
    all_d = np.zeros((H, W), np.float32)
    xi2 = ((verts[:, 0] - x_min_w) / res).astype(int)
    zi2 = ((verts[:, 2] - z_min_w) / res).astype(int)
    v2 = (xi2 >= 0) & (xi2 < W) & (zi2 >= 0) & (zi2 < H)
    np.add.at(all_d, (zi2[v2], xi2[v2]), 1)
    
    apt = (ndimage.gaussian_filter(all_d, 5) > 0.5).astype(np.uint8)
    apt = cv2.morphologyEx(apt, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
    apt = ndimage.binary_fill_holes(apt).astype(np.uint8)
    apt = cv2.erode(apt, np.ones((3,3), np.uint8))
    print(f"  Mask area: {apt.sum() * res * res:.1f}m²")
    
    # ═══ Phase 3: Hough wall selection ═══
    print("\n--- Phase 3: Hough wall selection ---")
    
    # Standard Hough on wall_final
    hough = cv2.HoughLines(wall_final, 1, np.pi/180, 50)
    if hough is None:
        print("ERROR: No Hough lines")
        return
    hough = hough[:, 0, :]
    
    # Filter to dominant angles
    filt = [(r, t) for r, t in hough if 
            min(abs(np.degrees(t)%180 - angle1), 180 - abs(np.degrees(t)%180 - angle1)) < 12 or
            min(abs(np.degrees(t)%180 - angle2), 180 - abs(np.degrees(t)%180 - angle2)) < 12]
    print(f"  {len(hough)} raw → {len(filt)} angle-filtered")
    
    def score_line(rho, theta):
        """Score wall line by density and max run along it within apartment mask."""
        ct, st = np.cos(theta), np.sin(theta)
        density = 0
        max_run = 0
        cur_run = 0
        in_apt = 0
        for t in range(-max(H, W), max(H, W)):
            x = int(rho * ct + t * (-st))
            y = int(rho * st + t * ct)
            if 0 <= x < W and 0 <= y < H:
                if apt[y, x]:
                    in_apt += 1
                    if wall_final[y, x] > 0:
                        density += 1
                        cur_run += 1
                        max_run = max(max_run, cur_run)
                    else:
                        cur_run = 0
        return density, max_run, in_apt
    
    # Separate families and find rho peaks
    def find_wall_peaks(lines, target_angle):
        fam = [(r, t) for r, t in lines if
               min(abs(np.degrees(t)%180 - target_angle), 180 - abs(np.degrees(t)%180 - target_angle)) < 12]
        if not fam:
            return []
        
        rhos = np.array([r for r, t in fam])
        rho_min, rho_max = rhos.min() - 5, rhos.max() + 5
        n_bins = max(int(rho_max - rho_min), 10)
        hist, bin_edges = np.histogram(rhos, bins=n_bins, range=(rho_min, rho_max))
        hs = ndimage.gaussian_filter1d(hist.astype(float), sigma=3)
        
        from scipy.signal import find_peaks
        pks, _ = find_peaks(hs, height=max(hs)*0.1, distance=int(0.3/res))  # 0.3m min between walls
        peak_rhos = bin_edges[pks] + 0.5
        
        # Score each peak
        walls = []
        avg_theta = np.mean([t for r, t in fam])
        for pr in peak_rhos:
            d, mr, ia = score_line(pr, avg_theta)
            score = d * np.sqrt(mr + 1)
            walls.append((pr, avg_theta, score, d, mr, ia))
        
        walls.sort(key=lambda x: x[2], reverse=True)
        return walls
    
    walls1 = find_wall_peaks(filt, angle1)
    walls2 = find_wall_peaks(filt, angle2)
    
    print(f"  Fam1 ({angle1:.0f}°): {len(walls1)} peaks")
    for r, t, s, d, mr, ia in walls1[:8]:
        print(f"    ρ={r:.0f}, score={s:.0f} (density={d}, run={mr})")
    print(f"  Fam2 ({angle2:.0f}°): {len(walls2)} peaks")
    for r, t, s, d, mr, ia in walls2[:8]:
        print(f"    ρ={r:.0f}, score={s:.0f} (density={d}, run={mr})")
    
    # Select walls: use mean score threshold
    def select_walls(walls, max_n=4):
        if not walls: return []
        scores = [s for _, _, s, _, _, _ in walls[:max_n*2]]
        mean_s = np.mean(scores)
        sel = []
        for w in walls:
            if w[2] >= mean_s * 0.5 and len(sel) < max_n:
                sel.append(w)
            elif len(sel) < 2:
                sel.append(w)
        return sel
    
    sel1 = select_walls(walls1, max_n=4)
    sel2 = select_walls(walls2, max_n=4)
    all_selected = sel1 + sel2
    
    print(f"\n  Selected: {len(sel1)} + {len(sel2)} = {len(all_selected)} walls")
    
    # ═══ Phase 4: Draw walls + partition ═══
    print("\n--- Phase 4: Draw walls + flood fill ---")
    
    # Draw selected wall lines on apartment mask
    partition = apt.copy() * 255
    
    for rho, theta, score, d, mr, ia in all_selected:
        ct, st = np.cos(theta), np.sin(theta)
        # Draw line across image (thickness=3 for solid wall)
        x0 = rho * ct
        y0 = rho * st
        dx, dy = -st * 600, ct * 600
        p1 = (int(x0 - dx), int(y0 - dy))
        p2 = (int(x0 + dx), int(y0 + dy))
        cv2.line(partition, p1, p2, 0, thickness=3)
    
    # Also cut by apartment boundary (outside = 0)
    partition[apt == 0] = 0
    
    # Flood fill → connected components
    n_labels, labels = cv2.connectedComponents(partition)
    
    rooms = []
    for i in range(1, n_labels):
        area_px = (labels == i).sum()
        area_m2 = area_px * res * res
        if area_m2 < 1.5:
            continue
        ys, xs = np.where(labels == i)
        cx_px, cy_px = xs.mean(), ys.mean()
        bbox_w = (xs.max() - xs.min()) * res
        bbox_h = (ys.max() - ys.min()) * res
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)
        
        rooms.append({
            'label': i, 'area': area_m2,
            'cx': float(cx_px), 'cy': float(cy_px),
            'aspect': float(aspect)
        })
    
    rooms.sort(key=lambda r: r['area'], reverse=True)
    total = sum(r['area'] for r in rooms)
    
    print(f"  Initial: {len(rooms)} rooms, {total:.1f}m²")
    
    # ═══ Phase 5: Wall recovery for oversized rooms ═══
    MAX_ROOM = 11.0
    rejected1 = [w for w in walls1 if w not in sel1]
    rejected2 = [w for w in walls2 if w not in sel2]
    
    for attempt in range(3):
        oversized = [r for r in rooms if r['area'] > MAX_ROOM]
        if not oversized:
            break
        
        print(f"\n  Recovery {attempt+1}: {len(oversized)} oversized rooms")
        
        best = None
        best_score = 0
        for r in oversized:
            mask = (labels == r['label'])
            for wl in [rejected1, rejected2]:
                for w in wl:
                    rho, theta = w[0], w[1]
                    # Check if line crosses this room
                    ct, st = np.cos(theta), np.sin(theta)
                    crosses = 0
                    for t in range(-max(H, W), max(H, W), 3):
                        x = int(rho * ct + t * (-st))
                        y = int(rho * st + t * ct)
                        if 0 <= x < W and 0 <= y < H and mask[y, x]:
                            crosses += 1
                    if crosses > 10 and w[2] > best_score:
                        best = w
                        best_score = w[2]
        
        if best is None:
            break
        
        rho, theta = best[0], best[1]
        print(f"  Adding wall: ρ={rho:.0f}, score={best[2]:.0f}")
        all_selected.append(best)
        
        # Remove from rejected
        for wl in [rejected1, rejected2]:
            if best in wl:
                wl.remove(best)
        
        # Redraw and re-partition
        partition = apt.copy() * 255
        for rho2, theta2, *_ in all_selected:
            ct, st = np.cos(theta2), np.sin(theta2)
            x0 = rho2 * ct
            y0 = rho2 * st
            dx, dy = -st * 600, ct * 600
            cv2.line(partition, (int(x0-dx), int(y0-dy)), (int(x0+dx), int(y0+dy)), 0, 3)
        partition[apt == 0] = 0
        
        n_labels, labels = cv2.connectedComponents(partition)
        rooms = []
        for i in range(1, n_labels):
            area_px = (labels == i).sum()
            area_m2 = area_px * res * res
            if area_m2 < 1.5:
                continue
            ys, xs = np.where(labels == i)
            rooms.append({
                'label': i, 'area': area_m2,
                'cx': float(xs.mean()), 'cy': float(ys.mean()),
                'aspect': float(max((xs.max()-xs.min()), 1) * res / max((ys.max()-ys.min()), 1) / res)
            })
        rooms.sort(key=lambda r: r['area'], reverse=True)
        total = sum(r['area'] for r in rooms)
    
    # ═══ Phase 6: Smart merge of tiny rooms ═══
    # Merge rooms < 2m² with their largest neighbor
    while True:
        tiny = [r for r in rooms if r['area'] < 2.0]
        if not tiny:
            break
        r = tiny[0]
        mask = (labels == r['label'])
        # Find adjacent room labels (dilate and check)
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((5,5), np.uint8))
        neighbors = set(labels[dilated > 0]) - {0, r['label']}
        
        if not neighbors:
            rooms.remove(r)
            continue
        
        # Merge into largest neighbor
        best_neighbor = max(neighbors, key=lambda n: (labels == n).sum())
        labels[mask] = best_neighbor
        
        # Update room data
        rooms.remove(r)
        for room in rooms:
            if room['label'] == best_neighbor:
                ys, xs = np.where(labels == best_neighbor)
                room['area'] = len(xs) * res * res
                room['cx'] = float(xs.mean())
                room['cy'] = float(ys.mean())
                break
    
    total = sum(r['area'] for r in rooms)
    
    # Classify rooms
    for r in rooms:
        a = r['area']
        asp = r['aspect']
        if asp > 2.5 and a < 8:
            r['name'] = 'Hallway'
        elif a > 8:
            r['name'] = 'Room'
        elif a > 4:
            r['name'] = 'Bathroom'
        else:
            r['name'] = 'Closet'
    
    # Deduplicate names
    counts = {}
    for r in rooms:
        n = r['name']
        counts[n] = counts.get(n, 0) + 1
        if counts[n] > 1:
            r['name'] = f"{n} {counts[n]}"
    
    print(f"\n  Final: {len(rooms)} rooms, {total:.1f}m²")
    for r in rooms:
        print(f"    {r['name']}: {r['area']:.1f}m²")
    
    # ═══ Render ═══
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].set_title('Cross-Section + Density')
    axes[0,0].imshow(wall_img, cmap='gray_r', origin='lower')
    
    axes[0,1].set_title(f'After Bridging ({n_bridges} bridges)')
    axes[0,1].imshow(bridged, cmap='gray_r', origin='lower')
    
    axes[0,2].set_title(f'Final Wall Image ({(wall_final>0).mean()*100:.1f}%)')
    axes[0,2].imshow(wall_final, cmap='gray_r', origin='lower')
    
    # Wall lines on image
    ax = axes[1,0]
    ax.set_title(f'Selected Wall Lines ({len(all_selected)})')
    ax.imshow(wall_final, cmap='gray', origin='lower', alpha=0.4)
    for rho, theta, *_ in all_selected:
        ct, st = np.cos(theta), np.sin(theta)
        x0, y0 = rho*ct, rho*st
        dx, dy = -st*500, ct*500
        ax.plot([x0-dx, x0+dx], [y0-dy, y0+dy], 'r-', lw=1.5)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    
    axes[1,1].set_title('Partition Image')
    axes[1,1].imshow(partition, cmap='gray', origin='lower')
    
    # Room map
    ax = axes[1,2]
    colors = plt.cm.Pastel1(np.linspace(0, 1, max(len(rooms), 1)))
    ax.set_title(f'v57e — {len(rooms)} rooms, {total:.1f}m²\nAngles: {angle1:.0f}°, {angle2:.0f}°')
    
    room_img = np.ones((H, W, 3))
    room_img[apt == 0] = [0.95, 0.95, 0.95]
    for i, r in enumerate(rooms):
        mask = labels == r['label']
        room_img[mask] = colors[i % len(colors)][:3]
    room_img[wall_final > 0] = [0.15, 0.15, 0.15]
    ax.imshow(room_img, origin='lower')
    
    for r in rooms:
        ax.text(r['cx'], r['cy'], f"{r['name']}\n{r['area']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))
    
    bar_px = 1.0 / res
    ax.plot([10, 10+bar_px], [15, 15], 'k-', lw=3)
    ax.text(10+bar_px/2, 30, '1m', ha='center', fontsize=9)
    
    for a in axes.flat:
        a.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir / 'floorplan.png'}")
    
    # JSON
    result = {
        'version': 'v57e',
        'angles': [float(angle1), float(angle2)],
        'walls': len(all_selected),
        'rooms': [{k: v for k, v in r.items() if k != 'label'} for r in rooms],
        'total_area': float(total)
    }
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
