#!/usr/bin/env python3
"""
mesh2plan v60d - Rotated Rectangles with Mutual Clipping

1. Detect 5 room seeds via wall density + watershed
2. Fit rotated bounding rectangle for each room
3. For each pair of overlapping rectangles, clip by half-plane at overlap midline
4. Result: non-overlapping rooms with clean angled walls
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           label, binary_closing, binary_opening, gaussian_filter)
from skimage.measure import find_contours
from skimage.segmentation import watershed
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, split
from shapely.affinity import rotate
import argparse


RESOLUTION = 0.02
WALL_CLOSE_RADIUS = 25
MIN_ROOM_AREA = 2.0
DENSITY_PCT = 60


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    print(f"Loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh


def wall_density_image(mesh, res):
    normals = mesh.face_normals
    wm = np.abs(normals[:, 1]) < 0.5
    centroids = mesh.vertices[mesh.faces[wm]].mean(axis=1)
    areas = mesh.area_faces[wm]
    xs, zs = centroids[:, 0], centroids[:, 2]
    xmin, xmax = xs.min()-0.2, xs.max()+0.2
    zmin, zmax = zs.min()-0.2, zs.max()+0.2
    w = int((xmax-xmin)/res)+1
    h = int((zmax-zmin)/res)+1
    d = np.zeros((h,w), dtype=np.float64)
    xi = ((xs-xmin)/res).astype(int).clip(0,w-1)
    zi = ((zs-zmin)/res).astype(int).clip(0,h-1)
    np.add.at(d, (zi,xi), areas)
    return d, (xmin,zmin)


def get_apt_mask(mesh, res, origin, shape):
    v = mesh.vertices
    xmin,zmin = origin
    h,w = shape
    m = np.zeros((h,w), dtype=bool)
    xi = ((v[:,0]-xmin)/res).astype(int).clip(0,w-1)
    zi = ((v[:,2]-zmin)/res).astype(int).clip(0,h-1)
    m[zi,xi] = True
    m = binary_dilation(m, iterations=5)
    m = binary_fill_holes(m)
    m = binary_erosion(m, iterations=5)
    return m


def detect_angles(mesh):
    normals = mesh.face_normals
    areas = mesh.area_faces
    wm = np.abs(normals[:,1]) < 0.5
    angles = np.degrees(np.arctan2(normals[wm][:,2], normals[wm][:,0])) % 180
    hist, bins = np.histogram(angles, bins=180, range=(0,180), weights=areas[wm])
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    hs = gaussian_filter1d(hist, sigma=3, mode='wrap')
    peaks, props = find_peaks(hs, distance=20, height=0)
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props['peak_heights'])[-2:]]
        wa = sorted([(bins[p]+90)%180 for p in top2])
    else:
        wa = [30,120]
    print(f"Wall angles: {wa[0]:.0f}° and {wa[1]:.0f}°")
    return wa


def get_room_seeds(density, apt_mask, res):
    nonzero = density[density>0]
    thresh = np.percentile(nonzero, DENSITY_PCT)
    wm = (density >= thresh) & apt_mask
    wm = binary_dilation(wm, iterations=2)
    from skimage.morphology import disk as sk_disk
    for angle in [29, 119]:
        rad = np.radians(angle)
        length = WALL_CLOSE_RADIUS
        struct = np.zeros((2*length+1, 2*length+1), dtype=bool)
        for t in np.linspace(-length, length, 4*length+1):
            r = int(round(length+t*np.sin(rad)))
            c = int(round(length+t*np.cos(rad)))
            if 0<=r<2*length+1 and 0<=c<2*length+1:
                struct[r,c] = True
        wm = binary_closing(wm, structure=struct)
    wm = binary_closing(wm, structure=sk_disk(5))
    wm = binary_opening(wm, iterations=2)
    interior = apt_mask & ~wm
    labeled, n = label(interior)
    rooms = [(i, (labeled==i).sum()*res*res) for i in range(1,n+1) if (labeled==i).sum()*res*res >= MIN_ROOM_AREA]
    
    gradient = gaussian_filter(density, sigma=2)
    gradient /= gradient.max()+1e-10
    markers = np.zeros_like(labeled)
    for rid,_ in rooms: markers[labeled==rid] = rid
    expanded = watershed(gradient, markers=markers, mask=apt_mask)
    new_rooms = [(rid, (expanded==rid).sum()*res*res) for rid,_ in rooms if (expanded==rid).sum()*res*res >= MIN_ROOM_AREA]
    print(f"Rooms: {len(new_rooms)}, {sum(a for _,a in new_rooms):.1f}m²")
    return expanded, new_rooms, wm


def fit_rotated_rect(coords, angle):
    coords = np.array(coords)
    rad = np.radians(-angle)
    c,s = np.cos(rad), np.sin(rad)
    cx,cy = coords.mean(axis=0)
    rot = np.column_stack([
        (coords[:,0]-cx)*c - (coords[:,1]-cy)*s,
        (coords[:,0]-cx)*s + (coords[:,1]-cy)*c
    ])
    xmin,xmax = rot[:,0].min(), rot[:,0].max()
    ymin,ymax = rot[:,1].min(), rot[:,1].max()
    corners_rot = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    rb = np.radians(angle)
    cb,sb = np.cos(rb), np.sin(rb)
    return Polygon([(rx*cb-ry*sb+cx, rx*sb+ry*cb+cy) for rx,ry in corners_rot])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('-o', '--output', default='v60d_clipped_rects.png')
    args = parser.parse_args()
    
    mesh = load_mesh(args.mesh)
    wall_angles = detect_angles(mesh)
    density, origin = wall_density_image(mesh, RESOLUTION)
    apt_mask = get_apt_mask(mesh, RESOLUTION, origin, density.shape)
    expanded, rooms, wall_mask = get_room_seeds(density, apt_mask, RESOLUTION)
    
    xmin, zmin = origin
    
    # Get room contours and fit rectangles
    room_rects = []
    room_centroids = []
    
    for rid, area in rooms:
        rm = (expanded == rid).astype(float)
        contours = find_contours(rm, 0.5)
        if not contours:
            room_rects.append(None)
            room_centroids.append((0,0))
            continue
        contour = max(contours, key=len)
        coords = [(c[1]*RESOLUTION+xmin, c[0]*RESOLUTION+zmin) for c in contour]
        
        rect = fit_rotated_rect(coords, wall_angles[0])
        room_rects.append(rect)
        room_centroids.append((rect.centroid.x, rect.centroid.y))
    
    # Mutual clipping: for each pair of overlapping rooms, 
    # cut each room by a line at the dominant angle through the overlap center
    clipped = list(room_rects)
    
    for i in range(len(clipped)):
        if clipped[i] is None: continue
        for j in range(i+1, len(clipped)):
            if clipped[j] is None: continue
            
            overlap = clipped[i].intersection(clipped[j])
            if overlap.is_empty or overlap.area < 0.1:
                continue
            
            # Find the wall line: through overlap centroid, at the nearest dominant angle
            ocx, ocy = overlap.centroid.x, overlap.centroid.y
            
            # Determine which angle separates these rooms better
            ci = np.array(room_centroids[i])
            cj = np.array(room_centroids[j])
            diff = cj - ci
            room_angle = np.degrees(np.arctan2(diff[1], diff[0])) % 180
            
            # Wall should be perpendicular to the direction between room centers
            # Choose the dominant angle closest to perpendicular
            best_wall_angle = wall_angles[0]
            best_diff = 999
            for wa in wall_angles:
                # Wall at angle wa → perpendicular at wa+90
                perp = (wa + 90) % 180
                d = min(abs(room_angle - perp), 180 - abs(room_angle - perp))
                if d < best_diff:
                    best_diff = d
                    best_wall_angle = wa
            
            # Create a long cutting line at this angle through overlap centroid
            rad = np.radians(best_wall_angle)
            dx, dy = np.cos(rad)*50, np.sin(rad)*50
            cut_line = LineString([(ocx-dx, ocy-dy), (ocx+dx, ocy+dy)])
            
            # Split both rooms by this line
            try:
                parts_i = split(clipped[i], cut_line)
                parts_j = split(clipped[j], cut_line)
                
                # For room i, keep the part closest to room i's centroid
                if len(parts_i.geoms) >= 2:
                    clipped[i] = min(parts_i.geoms, 
                                     key=lambda g: g.centroid.distance(
                                         Polygon([room_centroids[i]]).centroid))
                
                # For room j, keep the part closest to room j's centroid
                if len(parts_j.geoms) >= 2:
                    clipped[j] = min(parts_j.geoms,
                                     key=lambda g: g.centroid.distance(
                                         Polygon([room_centroids[j]]).centroid))
            except Exception as e:
                print(f"  Clip failed ({i},{j}): {e}")
    
    # Also clip to apartment boundary
    apt_contours = find_contours(apt_mask.astype(float), 0.5)
    if apt_contours:
        bnd = max(apt_contours, key=len)
        bnd_coords = [(c[1]*RESOLUTION+xmin, c[0]*RESOLUTION+zmin) for c in bnd]
        apt_poly = Polygon(bnd_coords).simplify(0.1)
        if apt_poly.is_valid:
            for i in range(len(clipped)):
                if clipped[i] is not None:
                    c = clipped[i].intersection(apt_poly)
                    if isinstance(c, MultiPolygon):
                        c = max(c.geoms, key=lambda g: g.area)
                    if not c.is_empty:
                        clipped[i] = c
    
    # Report & plot
    print(f"\n=== RESULTS ===")
    total = 0
    pastel = ['#AEC6CF', '#FFB3BA', '#BAFFC9', '#FFFFBA', '#E8BAFF', '#FFD1A4']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Boundary
    if apt_contours:
        bnd = max(apt_contours, key=len)
        bx = [c[1]*RESOLUTION+xmin for c in bnd]
        by = [c[0]*RESOLUTION+zmin for c in bnd]
        ax.plot(bx, by, 'k-', linewidth=1, alpha=0.3)
    
    for idx, (poly, (rid, orig_area)) in enumerate(zip(clipped, rooms)):
        if poly is None or poly.is_empty:
            continue
        
        poly = poly.simplify(0.08)
        area = poly.area
        total += area
        
        bounds = poly.bounds
        w = bounds[2]-bounds[0]
        h = bounds[3]-bounds[1]
        aspect = max(w,h)/(min(w,h)+0.01)
        nv = len(poly.exterior.coords)-1
        
        if area > 8: lbl = "Room"
        elif area > 5: lbl = "Hallway" if aspect > 2.5 else "Room"
        elif area > 3: lbl = "Hallway" if aspect > 2 else "Bathroom"
        else: lbl = "Closet"
        
        color = pastel[idx % len(pastel)]
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=color, alpha=0.5)
        ax.plot(xs, ys, 'k-', linewidth=2.5)
        
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, f"{lbl}\n{area:.1f}m²", ha='center', va='center', fontsize=9, weight='bold')
        
        print(f"  {lbl}: {area:.1f}m² ({nv}v)")
    
    print(f"  Total: {total:.1f}m²")
    
    ax.set_title(f"v60d — {sum(1 for p in clipped if p and not p.is_empty)} rooms, {total:.1f}m²\nAngles: {wall_angles[0]:.0f}°, {wall_angles[1]:.0f}°")
    ax.set_aspect('equal')
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.plot([xlim[0]+0.3, xlim[0]+1.3], [ylim[0]+0.3, ylim[0]+0.3], 'k-', linewidth=3)
    ax.text(xlim[0]+0.8, ylim[0]+0.1, '1m', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
