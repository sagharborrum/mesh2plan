#!/usr/bin/env python3
"""
mesh2plan v55b - Cross-Section + Hough Line Graph

Combines two proven ideas:
- Cross-section slicing gives DIRECT wall geometry (no density estimation noise)
- Hough line graph (v48-v52) partitions space into rooms cleanly

Pipeline:
1. Take 10+ horizontal slices at wall height through mesh
2. Rasterize all cross-section segments → wall image
3. Overlay with v41b-style normal-filtered density for reinforcement
4. Apply v52-style Hough → wall selection → room partition
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_path')
    parser.add_argument('--output-dir', default='results/v55b')
    parser.add_argument('--resolution', type=float, default=0.02)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = args.resolution
    
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_path, process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"  {len(verts)} verts, Y: {verts[:,1].min():.2f} to {verts[:,1].max():.2f}")
    
    # ── Phase 1: Multi-slice cross-sections ──
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min
    # Slices at 30%-80% of height (wall region, skip floor/ceiling)
    slice_heights = np.linspace(y_min + 0.3*y_range, y_min + 0.8*y_range, 12)
    
    # Setup image coordinates
    x_vals, z_vals = verts[:, 0], verts[:, 2]
    x_min_w, x_max_w = x_vals.min() - 0.3, x_vals.max() + 0.3
    z_min_w, z_max_w = z_vals.min() - 0.3, z_vals.max() + 0.3
    W_img = int((x_max_w - x_min_w) / res) + 1
    H_img = int((z_max_w - z_min_w) / res) + 1
    
    xsection_density = np.zeros((H_img, W_img), dtype=np.float32)
    
    print(f"  Taking {len(slice_heights)} cross-sections...")
    total_segs = 0
    for y_h in slice_heights:
        try:
            section = mesh.section(plane_origin=[0, y_h, 0], plane_normal=[0, 1, 0])
            if section is None:
                continue
        except Exception:
            continue
        
        # Get 3D line segments directly
        # section.entities are line segments in 3D
        for entity in section.entities:
            pts = section.vertices[entity.points]
            # Draw each segment on the density image
            for i in range(len(pts) - 1):
                x0 = int((pts[i][0] - x_min_w) / res)
                z0 = int((pts[i][2] - z_min_w) / res)
                x1 = int((pts[i+1][0] - x_min_w) / res)
                z1 = int((pts[i+1][2] - z_min_w) / res)
                
                if (0 <= x0 < W_img and 0 <= z0 < H_img and
                    0 <= x1 < W_img and 0 <= z1 < H_img):
                    cv2.line(xsection_density, (x0, z0), (x1, z1), 1.0, thickness=1)
                    total_segs += 1
    
    print(f"  {total_segs} segments rasterized, image: {W_img}×{H_img}")
    print(f"  Cross-section wall coverage: {(xsection_density > 0).sum() / (W_img*H_img) * 100:.1f}%")
    
    # ── Phase 2: Normal-filtered wall density (v41b style) ──
    print("  Computing normal-filtered wall density...")
    if hasattr(mesh, 'face_normals'):
        normals = mesh.face_normals
        wall_mask_3d = np.abs(normals[:, 1]) < 0.5
        wall_centroids = verts[faces[wall_mask_3d]].mean(axis=1)
        
        wall_density = np.zeros((H_img, W_img), dtype=np.float32)
        xi = ((wall_centroids[:, 0] - x_min_w) / res).astype(int)
        zi = ((wall_centroids[:, 2] - z_min_w) / res).astype(int)
        valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
        np.add.at(wall_density, (zi[valid], xi[valid]), 1)
        
        print(f"  Wall faces: {wall_mask_3d.sum()}/{len(faces)} ({wall_mask_3d.mean()*100:.0f}%)")
    else:
        wall_density = np.zeros_like(xsection_density)
    
    # ── Phase 3: Fuse signals ──
    # Normalize each to [0, 1] then combine
    xs_norm = xsection_density / max(xsection_density.max(), 1)
    wd_norm = wall_density / max(np.percentile(wall_density[wall_density > 0], 95), 1) if wall_density.max() > 0 else wall_density
    wd_norm = np.minimum(wd_norm, 1.0)
    
    # Fused: cross-section is strong, wall density fills gaps
    fused = np.maximum(xs_norm * 2.0, wd_norm)
    fused = np.minimum(fused, 1.0)
    
    # To uint8 for Hough
    fused_u8 = (fused * 255).astype(np.uint8)
    
    # Binary wall mask
    thresh = max(1, int(np.percentile(fused_u8[fused_u8 > 0], 60)))
    wall_binary = (fused_u8 >= thresh).astype(np.uint8) * 255
    
    print(f"  Fused wall coverage: {(wall_binary > 0).sum() / (W_img*H_img) * 100:.1f}%")
    
    # ── Phase 4: Hough line detection ──
    lines = cv2.HoughLines(wall_binary, 1, np.pi/180, threshold=80)
    if lines is None:
        print("  No Hough lines!")
        return
    
    lines = lines[:, 0]
    print(f"  {len(lines)} Hough lines")
    
    # Cluster angles
    angles_deg = np.degrees(lines[:, 1]) % 180
    
    Z = linkage(angles_deg.reshape(-1, 1), method='complete')
    clusters = fcluster(Z, t=15, criterion='distance')
    
    families = {}
    for c in np.unique(clusters):
        mask = clusters == c
        families[c] = {
            'angle': np.median(angles_deg[mask]),
            'lines': lines[mask],
            'count': mask.sum()
        }
    
    sorted_fams = sorted(families.values(), key=lambda f: f['count'], reverse=True)[:2]
    fam0, fam1 = sorted_fams[0], sorted_fams[1]
    print(f"  Dominant angles: {fam0['angle']:.0f}° ({fam0['count']}), {fam1['angle']:.0f}° ({fam1['count']})")
    
    # ── Phase 5: Wall selection (v52 approach) ──
    all_walls = []
    for fam_idx, fam in enumerate(sorted_fams):
        angle_rad = np.radians(fam['angle'])
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        rhos = fam['lines'][:, 0]
        
        # Sort and cluster by rho
        sorted_rhos = np.sort(rhos)
        min_gap = int(0.35 / res)  # 0.35m min wall separation
        
        groups = []
        current = [sorted_rhos[0]]
        for r in sorted_rhos[1:]:
            if r - current[-1] < min_gap:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
        groups.append(current)
        
        for grp in groups:
            rho = np.mean(grp)
            votes = len(grp)
            
            # Score by density along line
            score = 0
            max_run = 0
            current_run = 0
            n = 0
            for t in range(-max(H_img, W_img), max(H_img, W_img), 2):
                px = int(rho * cos_t - t * sin_t)
                py = int(rho * sin_t + t * cos_t)
                if 0 <= px < W_img and 0 <= py < H_img:
                    val = fused[py, px]
                    score += val
                    n += 1
                    if val > 0.1:
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 0
            
            composite = score * np.sqrt(max(max_run, 1))
            offset_m = rho * res
            
            all_walls.append({
                'family': fam_idx,
                'angle': fam['angle'],
                'rho': rho,
                'votes': votes,
                'score': score,
                'max_run': max_run,
                'composite': composite,
                'offset_m': offset_m,
            })
    
    all_walls.sort(key=lambda w: w['composite'], reverse=True)
    
    # Select walls: above mean composite, min 3/family
    composites = [w['composite'] for w in all_walls]
    mean_comp = np.mean(composites)
    selected = [w for w in all_walls if w['composite'] > mean_comp * 0.6]
    
    for fam_idx in range(2):
        fam_sel = [w for w in selected if w['family'] == fam_idx]
        if len(fam_sel) < 3:
            extras = [w for w in all_walls if w['family'] == fam_idx and w not in selected]
            extras.sort(key=lambda w: w['composite'], reverse=True)
            selected.extend(extras[:3 - len(fam_sel)])
    
    # Deduplicate close walls within each family
    final_walls = []
    for fam_idx in range(2):
        fam_walls = sorted([w for w in selected if w['family'] == fam_idx], key=lambda w: w['rho'])
        if not fam_walls:
            continue
        deduped = [fam_walls[0]]
        for w in fam_walls[1:]:
            if abs(w['rho'] - deduped[-1]['rho']) > min_gap:
                deduped.append(w)
            elif w['composite'] > deduped[-1]['composite']:
                deduped[-1] = w
        final_walls.extend(deduped)
    
    print(f"  {len(final_walls)} walls selected:")
    for w in sorted(final_walls, key=lambda w: (w['family'], w['rho'])):
        print(f"    Fam{w['family']} ({w['angle']:.0f}°) rho={w['rho']:.0f} score={w['score']:.0f} run={w['max_run']} comp={w['composite']:.0f}")
    
    # ── Phase 6: Room partition ──
    # Apartment mask
    all_vert_density = np.zeros((H_img, W_img), dtype=np.float32)
    xi = ((verts[:, 0] - x_min_w) / res).astype(int)
    zi = ((verts[:, 2] - z_min_w) / res).astype(int)
    valid = (xi >= 0) & (xi < W_img) & (zi >= 0) & (zi < H_img)
    np.add.at(all_vert_density, (zi[valid], xi[valid]), 1)
    
    apt_mask = (ndimage.gaussian_filter(all_vert_density, sigma=3) > 0.5).astype(np.uint8)
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    apt_mask = ndimage.binary_fill_holes(apt_mask).astype(np.uint8)
    # Slight erosion to tighten boundary
    apt_mask = cv2.erode(apt_mask, np.ones((5, 5), np.uint8))
    apt_mask = cv2.morphologyEx(apt_mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    
    # Draw walls
    wall_img = np.zeros((H_img, W_img), dtype=np.uint8)
    for w in final_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        t = max(H_img, W_img) * 2
        x0 = int(rho * cos_t + t * sin_t)
        y0 = int(rho * sin_t - t * cos_t)
        x1 = int(rho * cos_t - t * sin_t)
        y1 = int(rho * sin_t + t * cos_t)
        cv2.line(wall_img, (x0, y0), (x1, y1), 255, thickness=3)
    
    wall_img &= (apt_mask * 255)
    
    # Add apartment boundary
    contours, _ = cv2.findContours(apt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(wall_img, contours, -1, 255, thickness=3)
    
    # Connected components
    interior = apt_mask.copy()
    interior[wall_img > 0] = 0
    n_labels, labels = cv2.connectedComponents(interior)
    
    rooms = []
    for i in range(1, n_labels):
        area_px = (labels == i).sum()
        area_m2 = area_px * res * res
        if area_m2 < 1.0:
            continue
        
        ys, xs = np.where(labels == i)
        cx_px, cy_px = xs.mean(), ys.mean()
        bbox_w = (xs.max() - xs.min()) * res
        bbox_h = (ys.max() - ys.min()) * res
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)
        
        if area_m2 > 8:
            name = "Room"
        elif aspect > 2.5:
            name = "Hallway"
        elif area_m2 < 3:
            name = "Closet"
        else:
            name = "Bathroom"
        
        rooms.append({
            'label': i, 'area_m2': area_m2,
            'cx_px': cx_px, 'cy_px': cy_px,
            'aspect': aspect, 'name': name,
        })
    
    rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # ── Phase 6b: Structural wall recovery (v52 style) ──
    # For rooms > 10m², check if a rejected wall can split them
    rejected = [w for w in all_walls if w not in final_walls]
    recovered = []
    for room in rooms:
        if room['area_m2'] < 10:
            continue
        room_mask = labels == room['label']
        best_wall = None
        best_split = 0
        for w in rejected:
            angle_rad = np.radians(w['angle'])
            rho = w['rho']
            cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
            
            # Check how many room pixels this wall crosses
            ys, xs = np.where(room_mask)
            # Distance of each pixel to the line
            dists = np.abs(xs * cos_t + ys * sin_t - rho)
            crossing = (dists < 3).sum()
            
            if crossing > best_split and crossing > 20:
                best_wall = w
                best_split = crossing
        
        if best_wall and best_wall not in recovered:
            recovered.append(best_wall)
            print(f"  Recovered wall for {room['name']} ({room['area_m2']:.1f}m²): "
                  f"Fam{best_wall['family']} rho={best_wall['rho']:.0f} crossing={best_split}")
    
    if recovered:
        final_walls.extend(recovered)
        # Redraw and repartition
        wall_img = np.zeros((H_img, W_img), dtype=np.uint8)
        for w in final_walls:
            angle_rad = np.radians(w['angle'])
            rho = w['rho']
            cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
            t = max(H_img, W_img) * 2
            x0 = int(rho * cos_t + t * sin_t)
            y0 = int(rho * sin_t - t * cos_t)
            x1 = int(rho * cos_t - t * sin_t)
            y1 = int(rho * sin_t + t * cos_t)
            cv2.line(wall_img, (x0, y0), (x1, y1), 255, thickness=3)
        
        wall_img &= (apt_mask * 255)
        if contours:
            cv2.drawContours(wall_img, contours, -1, 255, thickness=3)
        
        interior = apt_mask.copy()
        interior[wall_img > 0] = 0
        n_labels, labels = cv2.connectedComponents(interior)
        
        rooms = []
        for i in range(1, n_labels):
            area_px = (labels == i).sum()
            area_m2 = area_px * res * res
            if area_m2 < 1.0:
                continue
            ys, xs = np.where(labels == i)
            cx_px, cy_px = xs.mean(), ys.mean()
            bbox_w = (xs.max() - xs.min()) * res
            bbox_h = (ys.max() - ys.min()) * res
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)
            if area_m2 > 8: name = "Room"
            elif aspect > 2.5: name = "Hallway"
            elif area_m2 < 3: name = "Closet"
            else: name = "Bathroom"
            rooms.append({
                'label': i, 'area_m2': area_m2,
                'cx_px': cx_px, 'cy_px': cy_px,
                'aspect': aspect, 'name': name,
            })
        rooms.sort(key=lambda r: r['area_m2'], reverse=True)
    
    # Deduplicate names
    name_counts = {}
    for r in rooms:
        n = r['name']
        name_counts[n] = name_counts.get(n, 0) + 1
        if name_counts[n] > 1:
            r['name'] = f"{n} {name_counts[n]}"
    
    total = sum(r['area_m2'] for r in rooms)
    print(f"\n  {len(rooms)} rooms, {total:.1f}m² total")
    for r in rooms:
        print(f"    {r['name']}: {r['area_m2']:.1f}m² (aspect={r['aspect']:.1f})")
    
    # ── Phase 7: Render ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # 1. Cross-section density
    ax = axes[0]
    ax.set_title('Cross-Section Density')
    ax.imshow(xsection_density, cmap='hot', origin='lower')
    
    # 2. Fused wall image + Hough lines
    ax = axes[1]
    ax.set_title(f'Fused Walls + {len(final_walls)} Hough Lines')
    ax.imshow(fused, cmap='gray', origin='lower', alpha=0.6)
    for w in final_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        t = max(H_img, W_img)
        x0, y0 = rho*cos_t + t*sin_t, rho*sin_t - t*cos_t
        x1, y1 = rho*cos_t - t*sin_t, rho*sin_t + t*cos_t
        color = 'red' if w['family'] == 0 else 'blue'
        lw = 2 if w in recovered else 1
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.8)
    ax.set_xlim(0, W_img)
    ax.set_ylim(0, H_img)
    
    # 3. Room partition
    ax = axes[2]
    angles_str = f"{fam0['angle']:.0f}°, {fam1['angle']:.0f}°"
    ax.set_title(f'v55b Cross-Section + Hough\n{len(rooms)} rooms, {total:.1f}m², angles: {angles_str}')
    
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(rooms), 3)))
    room_img = np.ones((H_img, W_img, 3))
    for i, r in enumerate(rooms):
        mask = labels == r['label']
        room_img[mask] = colors[i % len(colors)][:3]
    
    ax.imshow(room_img, origin='lower')
    
    # Draw walls as double lines
    for w in final_walls:
        angle_rad = np.radians(w['angle'])
        rho = w['rho']
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        t = max(H_img, W_img)
        for dr in [-2, 2]:
            r = rho + dr
            x0, y0 = r*cos_t + t*sin_t, r*sin_t - t*cos_t
            x1, y1 = r*cos_t - t*sin_t, r*sin_t + t*cos_t
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=1.5)
    
    # Apartment boundary
    ax.contour(apt_mask, levels=[0.5], colors='black', linewidths=2, origin='lower')
    
    for r in rooms:
        ax.text(r['cx_px'], r['cy_px'], f"{r['name']}\n{r['area_m2']:.1f}m²",
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Scale bar
    bar_px = 1.0 / res
    ax.plot([10, 10+bar_px], [15, 15], 'k-', linewidth=3)
    ax.text(10+bar_px/2, 25, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'floorplan.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_dir / 'floorplan.png'}")


if __name__ == '__main__':
    main()
