// mesh2plan Web Worker — cross-section + wall analysis
// Receives triangle data, returns analysis results

self.onmessage = function(e) {
  const { triangles, meshBounds } = e.data;
  
  self.postMessage({ type: 'progress', msg: `Analyzing ${triangles.length} triangles...`, pct: 5 });
  
  const nSlices = 20;
  const yMin = meshBounds.min.y, yMax = meshBounds.max.y;
  const yRange = yMax - yMin;
  
  // Collect cross-section points
  const allXZ = [];
  const slices = [];
  
  for (let i = 0; i < nSlices; i++) {
    const y = yMin + yRange * (0.15 + 0.7 * i / (nSlices - 1));
    const pts = sliceAtY(triangles, y);
    if (pts.length < 4) continue;
    allXZ.push(...pts);
    slices.push({ height: y, points: pts, n: pts.length });
    self.postMessage({ type: 'progress', msg: `Slice ${i+1}/${nSlices}: ${pts.length} pts`, pct: 5 + 50 * (i+1)/nSlices });
  }
  
  if (allXZ.length < 50) {
    self.postMessage({ type: 'result', data: { walls: [], room: null, gaps: [], slices, angle: 0, allXZ } });
    return;
  }
  
  self.postMessage({ type: 'progress', msg: `Finding dominant angle (${allXZ.length} pts)...`, pct: 60 });
  
  const angle = findDominantAngle(allXZ);
  const angleRad = angle * Math.PI / 180;
  const rotated = allXZ.map(p => rotPt(p, -angleRad));
  
  self.postMessage({ type: 'progress', msg: `Finding walls at ${angle}°...`, pct: 80 });
  
  const xWalls = findWalls(rotated, 0);
  const zWalls = findWalls(rotated, 1);
  const allRaw = [...xWalls, ...zWalls];
  const boundary = filterBoundaryWalls(allRaw, rotated);
  self.postMessage({ type: 'progress', msg: `Filtered ${allRaw.length} → ${boundary.length} boundary walls`, pct: 85 });
  let walls = mergeWalls(boundary);
  
  // Convert coords back
  for (const w of walls) {
    if (w.axis === 'x') {
      w.startPt = rotPt([w.position, w.start], angleRad);
      w.endPt = rotPt([w.position, w.end], angleRad);
    } else {
      w.startPt = rotPt([w.start, w.position], angleRad);
      w.endPt = rotPt([w.end, w.position], angleRad);
    }
  }
  
  // Room polygon — build from wall intersections with rectilinear snapping
  self.postMessage({ type: 'progress', msg: 'Building room polygon...', pct: 90 });
  let room = buildRoomPolygon(walls, angleRad);
  
  // Gaps
  self.postMessage({ type: 'progress', msg: 'Detecting openings...', pct: 95 });
  const gaps = detectGaps(walls, rotated, angle);
  
  self.postMessage({ type: 'result', data: { walls, room, gaps, slices, angle, allXZ, rotated } });
};

function sliceAtY(tris, y) {
  const pts = [];
  for (const tri of tris) {
    const edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
    const hits = [];
    for (const [a, b] of edges) {
      if ((a[1] - y) * (b[1] - y) < 0) {
        const t = (y - a[1]) / (b[1] - a[1]);
        hits.push([a[0] + t*(b[0]-a[0]), a[2] + t*(b[2]-a[2])]);
      }
    }
    if (hits.length >= 2) { pts.push(hits[0], hits[1]); }
  }
  return pts;
}

function rotPt(p, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c*p[0] - s*p[1], s*p[0] + c*p[1]];
}

function findDominantAngle(points) {
  // Subsample for speed (use at most 5000 points)
  const step = Math.max(1, Math.floor(points.length / 5000));
  const sampled = [];
  for (let i = 0; i < points.length; i += step) sampled.push(points[i]);
  
  const xBuf = new Float64Array(sampled.length);
  const zBuf = new Float64Array(sampled.length);
  let best = 0, bestScore = 0;
  
  for (let deg = 0; deg < 180; deg += 2) { // Step by 2 for speed
    const rad = -deg * Math.PI / 180;
    const c = Math.cos(rad), s = Math.sin(rad);
    for (let i = 0; i < sampled.length; i++) {
      xBuf[i] = c*sampled[i][0] - s*sampled[i][1];
      zBuf[i] = s*sampled[i][0] + c*sampled[i][1];
    }
    const xh = histogramTyped(xBuf, 80);
    const zh = histogramTyped(zBuf, 80);
    let score = 0;
    for (let i = 0; i < 80; i++) score += xh[i]*xh[i] + zh[i]*zh[i];
    if (score > bestScore) { bestScore = score; best = deg; }
  }
  
  // Refine around best ±2°
  for (let deg = best-2; deg <= best+2; deg++) {
    const rad = -deg * Math.PI / 180;
    const c = Math.cos(rad), s = Math.sin(rad);
    for (let i = 0; i < sampled.length; i++) {
      xBuf[i] = c*sampled[i][0] - s*sampled[i][1];
      zBuf[i] = s*sampled[i][0] + c*sampled[i][1];
    }
    const xh = histogramTyped(xBuf, 80);
    const zh = histogramTyped(zBuf, 80);
    let score = 0;
    for (let i = 0; i < 80; i++) score += xh[i]*xh[i] + zh[i]*zh[i];
    if (score > bestScore) { bestScore = score; best = deg; }
  }
  
  return ((best % 180) + 180) % 180;
}

function histogramTyped(values, nBins) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < values.length; i++) { if (values[i] < mn) mn = values[i]; if (values[i] > mx) mx = values[i]; }
  const range = mx-mn||1;
  const bins = new Int32Array(nBins);
  for (let i = 0; i < values.length; i++) bins[Math.min(nBins-1, Math.floor((values[i]-mn)/range*nBins))]++;
  return bins;
}

function histogram(values, nBins) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < values.length; i++) { if (values[i] < mn) mn = values[i]; if (values[i] > mx) mx = values[i]; }
  const range = mx-mn||1;
  const bins = new Array(nBins).fill(0);
  for (let i = 0; i < values.length; i++) bins[Math.min(nBins-1, Math.floor((values[i]-mn)/range*nBins))]++;
  return bins;
}

function findWalls(rotated, axis, minInliers=10, distThresh=0.04) {
  const coords = rotated.map(p => p[axis]);
  const other = rotated.map(p => p[1-axis]);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < coords.length; i++) { if (coords[i] < mn) mn = coords[i]; if (coords[i] > mx) mx = coords[i]; }
  const nBins = Math.max(40, Math.floor((mx-mn)/0.02));
  const binW = (mx-mn)/nBins;
  const hist = new Array(nBins).fill(0);
  for (const c of coords) hist[Math.min(nBins-1, Math.floor((c-mn)/(mx-mn)*nBins))]++;
  
  const sorted = [...hist].sort((a,b) => a-b);
  const median = sorted[Math.floor(nBins/2)];
  const threshold = Math.max(median * 3, minInliers);
  
  const walls = [];
  let inPeak = false, pw = 0, ps = 0;
  for (let i = 0; i <= nBins; i++) {
    const bc = mn + (i + 0.5) * binW;
    if (i < nBins && hist[i] > threshold) {
      if (!inPeak) { inPeak = true; pw = 0; ps = 0; }
      pw += hist[i]; ps += bc * hist[i];
    } else if (inPeak) {
      const wallPos = ps / pw;
      const nearPts = [];
      for (let j = 0; j < coords.length; j++) {
        if (Math.abs(coords[j] - wallPos) < distThresh * 2) nearPts.push(other[j]);
      }
      if (nearPts.length >= minInliers) {
        nearPts.sort((a,b) => a-b);
        const start = nearPts[Math.floor(nearPts.length * 0.02)];
        const end = nearPts[Math.floor(nearPts.length * 0.98)];
        if (end - start > 0.3) {
          walls.push({ axis: axis === 0 ? 'x' : 'z', position: wallPos, start, end, length: end-start, nPoints: nearPts.length });
        }
      }
      inPeak = false;
    }
  }
  return walls;
}

function filterBoundaryWalls(walls, rotated) {
  // Real walls are at room boundaries — points primarily on one side
  // Furniture creates peaks with points on both sides
  return walls.filter(w => {
    const axis = w.axis === 'x' ? 0 : 1;
    const pos = w.position;
    const band = 0.3; // check 30cm on each side
    let below = 0, above = 0;
    for (const p of rotated) {
      const d = p[axis] - pos;
      if (d < -0.06 && d > -band) below++;
      else if (d > 0.06 && d < band) above++;
    }
    // A real boundary wall has asymmetric density — most points on one side
    const total = below + above;
    if (total < 20) return true; // not enough data, keep it
    const ratio = Math.min(below, above) / Math.max(below, above);
    // ratio < 0.4 means mostly one-sided (wall), > 0.6 means both sides (furniture)
    return ratio < 0.5;
  });
}

function mergeWalls(walls, dist=0.15) {
  const merged = []; const used = new Set();
  for (let i = 0; i < walls.length; i++) {
    if (used.has(i)) continue;
    const group = [walls[i]];
    for (let j = i+1; j < walls.length; j++) {
      if (used.has(j) || walls[i].axis !== walls[j].axis) continue;
      if (Math.abs(walls[i].position - walls[j].position) < dist) { group.push(walls[j]); used.add(j); }
    }
    const tp = group.reduce((s,g) => s+g.nPoints, 0);
    const ap = group.reduce((s,g) => s+g.position*g.nPoints, 0)/tp;
    merged.push({ ...group[0], position: ap, start: Math.min(...group.map(g=>g.start)),
      end: Math.max(...group.map(g=>g.end)), length: Math.max(...group.map(g=>g.end))-Math.min(...group.map(g=>g.start)), nPoints: tp });
    used.add(i);
  }
  return merged;
}

function detectGaps(walls, rotated, angleDeg) {
  const angleRad = angleDeg * Math.PI / 180;
  const gaps = [];
  for (const w of walls) {
    const axis = w.axis === 'x' ? 0 : 1;
    const near = [];
    for (const p of rotated) if (Math.abs(p[axis] - w.position) < 0.06) near.push(p[1-axis]);
    if (near.length < 10) continue;
    near.sort((a,b) => a-b);
    for (let i = 0; i < near.length - 1; i++) {
      const gap = near[i+1] - near[i];
      if (gap > 0.3) {
        const gMid = (near[i]+near[i+1])/2;
        let p1, p2, mid;
        if (w.axis === 'x') { p1 = rotPt([w.position, near[i]], angleRad); p2 = rotPt([w.position, near[i+1]], angleRad); mid = rotPt([w.position, gMid], angleRad); }
        else { p1 = rotPt([near[i], w.position], angleRad); p2 = rotPt([near[i+1], w.position], angleRad); mid = rotPt([gMid, w.position], angleRad); }
        gaps.push({ type: gap > 0.6 && gap < 1.3 ? 'door' : (gap < 2.0 ? 'window' : 'opening'), width: gap, start: p1, end: p2, mid });
      }
    }
  }
  return gaps;
}

function convexHull2D(points) {
  if (points.length < 3) return points;
  const pts = [...points].sort((a,b) => a[0]-b[0] || a[1]-b[1]);
  const cross = (o,a,b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lower = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop(); lower.push(p); }
  const upper = [];
  for (const p of pts.reverse()) { while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop(); upper.push(p); }
  return lower.slice(0,-1).concat(upper.slice(0,-1));
}

function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) { const j = (i+1)%pts.length; a += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]; }
  return Math.abs(a)/2;
}

function polygonPerimeter(pts) {
  let p = 0;
  for (let i = 0; i < pts.length; i++) { const j = (i+1)%pts.length; p += Math.sqrt((pts[j][0]-pts[i][0])**2 + (pts[j][1]-pts[i][1])**2); }
  return p;
}

function buildRoomPolygon(walls, angleRad) {
  const xW = walls.filter(w => w.axis === 'x').sort((a,b) => a.position - b.position);
  const zW = walls.filter(w => w.axis === 'z').sort((a,b) => a.position - b.position);
  
  if (xW.length < 2 || zW.length < 2) {
    // Fallback: convex hull of all wall endpoints
    const allPts = [];
    for (const w of walls) { allPts.push(w.startPt, w.endPt); }
    if (allPts.length < 3) return null;
    const hull = convexHull2D(allPts);
    if (hull.length < 3) return null;
    return { exterior: [...hull, hull[0]], area: polygonArea(hull), perimeter: polygonPerimeter(hull) };
  }
  
  // Find all valid intersections between perpendicular walls
  const intersections = [];
  for (const xw of xW) {
    for (const zw of zW) {
      // Check overlap: x-wall spans z from xw.start to xw.end
      // z-wall spans x from zw.start to zw.end
      const ext = 0.3; // extension tolerance
      const zInRange = xw.start - ext <= zw.position && zw.position <= xw.end + ext;
      const xInRange = zw.start - ext <= xw.position && xw.position <= zw.end + ext;
      if (zInRange && xInRange) {
        intersections.push({
          rotPt: [xw.position, zw.position],
          origPt: rotPt([xw.position, zw.position], angleRad),
          xWall: xw,
          zWall: zw,
        });
      }
    }
  }
  
  if (intersections.length < 3) {
    const allPts = [];
    for (const w of walls) { allPts.push(w.startPt, w.endPt); }
    if (allPts.length < 3) return null;
    const hull = convexHull2D(allPts);
    if (hull.length < 3) return null;
    return { exterior: [...hull, hull[0]], area: polygonArea(hull), perimeter: polygonPerimeter(hull) };
  }
  
  // Try to walk the boundary — find the outermost polygon
  // Use convex hull of intersections as the room boundary
  const pts = intersections.map(i => i.origPt);
  const hull = convexHull2D(pts);
  
  if (hull.length < 3) return null;
  
  // Now try to snap hull vertices to actual wall intersections and create axis-aligned segments
  // This gives us a clean rectilinear polygon
  const snappedHull = snapToAxisAligned(hull, intersections, angleRad);
  
  const exterior = [...snappedHull, snappedHull[0]];
  return {
    exterior,
    area: polygonArea(snappedHull),
    perimeter: polygonPerimeter(snappedHull),
  };
}

function snapToAxisAligned(hull, intersections, angleRad) {
  const negRad = -angleRad;
  const hullRot = hull.map(p => rotPt(p, negRad));
  const intRot = intersections.map(i => i.rotPt);
  
  // Snap hull vertices to nearest intersection
  const snapped = hullRot.map(hp => {
    let bestDist = Infinity, bestPt = hp;
    for (const ip of intRot) {
      const d = Math.sqrt((hp[0]-ip[0])**2 + (hp[1]-ip[1])**2);
      if (d < bestDist) { bestDist = d; bestPt = ip; }
    }
    return bestDist < 0.5 ? [...bestPt] : [...hp];
  });
  
  // Remove duplicate consecutive vertices
  const deduped = [snapped[0]];
  for (let i = 1; i < snapped.length; i++) {
    const prev = deduped[deduped.length - 1];
    if (Math.abs(snapped[i][0]-prev[0]) > 0.01 || Math.abs(snapped[i][1]-prev[1]) > 0.01) {
      deduped.push(snapped[i]);
    }
  }
  
  // Build rectilinear path: insert corners where edges aren't axis-aligned
  const rectilinear = [];
  for (let i = 0; i < deduped.length; i++) {
    const curr = deduped[i];
    const next = deduped[(i + 1) % deduped.length];
    rectilinear.push(curr);
    
    const dx = Math.abs(next[0] - curr[0]);
    const dz = Math.abs(next[1] - curr[1]);
    if (dx > 0.1 && dz > 0.1) {
      // Need a corner — try both L-turn options
      const opt1 = [next[0], curr[1]];
      const opt2 = [curr[0], next[1]];
      
      // Score each option: prefer the one that:
      // 1. Is near an actual intersection
      // 2. Doesn't make the polygon self-intersecting
      let score1 = Infinity, score2 = Infinity;
      for (const ip of intRot) {
        const d1 = Math.sqrt((opt1[0]-ip[0])**2 + (opt1[1]-ip[1])**2);
        const d2 = Math.sqrt((opt2[0]-ip[0])**2 + (opt2[1]-ip[1])**2);
        if (d1 < score1) score1 = d1;
        if (d2 < score2) score2 = d2;
      }
      
      // Also prefer corners that keep the polygon convex-ish
      // (check if the turn direction matches the hull winding)
      rectilinear.push(score1 < score2 ? opt1 : opt2);
    }
  }
  
  return rectilinear.map(p => rotPt(p, angleRad));
}
