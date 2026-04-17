#!/usr/bin/env python3
"""Generate a flat disk with approximately uniform (Delaunay) triangulation.

Uses Poisson-disk sampling to place points, then Delaunay triangulation,
then clips to the disk boundary. Produces much more isotropic triangles
than the concentric-ring approach, avoiding the radial fold lines.

Usage:
    python3 scripts/gen_uniform_disk.py [--target-verts N] [--radius R] [--out PATH]
"""

import argparse
import math
import os
import random


def poisson_disk_sample(radius, min_dist, seed=42):
    """Simple Poisson-disk sampling in a disk of given radius."""
    random.seed(seed)
    points = [(0.0, 0.0)]  # always include center
    active = [0]
    k = 30  # candidates per point

    while active:
        idx = random.choice(active)
        px, py = points[idx]
        found = False
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(min_dist, 2 * min_dist)
            nx = px + dist * math.cos(angle)
            ny = py + dist * math.sin(angle)
            if nx * nx + ny * ny > radius * radius:
                continue
            too_close = False
            for (qx, qy) in points:
                if (nx - qx) ** 2 + (ny - qy) ** 2 < min_dist * min_dist:
                    too_close = True
                    break
            if not too_close:
                points.append((nx, ny))
                active.append(len(points) - 1)
                found = True
        if not found:
            active.remove(idx)
    return points


def delaunay_triangulate(points):
    """Bowyer-Watson Delaunay triangulation."""
    # Super-triangle enclosing all points
    big = 10.0
    st = [(-big, -big), (big, -big), (0, big)]
    n_st = len(points)
    all_pts = list(points) + st

    triangles = [(n_st, n_st + 1, n_st + 2)]

    for i in range(len(points)):
        px, py = all_pts[i]
        bad = []
        for tri in triangles:
            if in_circumcircle(all_pts, tri, px, py):
                bad.append(tri)

        # Find boundary polygon of the hole
        edges = []
        for tri in bad:
            for j in range(3):
                e = (tri[j], tri[(j + 1) % 3])
                # Edge is on boundary if it's not shared by another bad triangle
                shared = False
                for other in bad:
                    if other == tri:
                        continue
                    if e[0] in other and e[1] in other:
                        shared = True
                        break
                if not shared:
                    edges.append(e)

        for tri in bad:
            triangles.remove(tri)

        for e in edges:
            triangles.append((i, e[0], e[1]))

    # Remove triangles that reference super-triangle vertices
    result = []
    for tri in triangles:
        if tri[0] >= n_st or tri[1] >= n_st or tri[2] >= n_st:
            continue
        result.append(tri)
    return result


def in_circumcircle(pts, tri, px, py):
    ax, ay = pts[tri[0]]
    bx, by = pts[tri[1]]
    cx, cy = pts[tri[2]]
    dx = ax - px; dy = ay - py
    ex = bx - px; ey = by - py
    fx = cx - px; fy = cy - py
    det = (dx * (ey * (fx*fx + fy*fy) - fy * (ex*ex + ey*ey))
         - dy * (ex * (fx*fx + fy*fy) - fx * (ex*ex + ey*ey))
         + (dx*dx + dy*dy) * (ex * fy - ey * fx))
    # Check orientation
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if orient > 0:
        return det > 0
    else:
        return det < 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-verts", type=int, default=200,
                    help="approximate number of vertices (default 200)")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="disk radius (default 1.0)")
    ap.add_argument("--out", default="meshes/flat_disk.obj",
                    help="output OBJ path")
    args = ap.parse_args()

    R = args.radius
    # Estimate min_dist from target vertex count: area = πR², each point
    # occupies ~(min_dist)² area in a Poisson distribution.
    area = math.pi * R * R
    min_dist = math.sqrt(area / args.target_verts) * 0.85

    print(f"Sampling with min_dist={min_dist:.4f} for ~{args.target_verts} vertices...")
    points_2d = poisson_disk_sample(R, min_dist)

    # Add boundary points to ensure clean edge
    n_boundary = max(16, int(2 * math.pi * R / min_dist))
    for i in range(n_boundary):
        angle = 2 * math.pi * i / n_boundary
        bx = R * math.cos(angle)
        by = R * math.sin(angle)
        # Check not too close to existing points
        too_close = False
        for (qx, qy) in points_2d:
            if (bx - qx) ** 2 + (by - qy) ** 2 < (min_dist * 0.5) ** 2:
                too_close = True
                break
        if not too_close:
            points_2d.append((bx, by))

    print(f"Got {len(points_2d)} vertices, triangulating...")
    tris = delaunay_triangulate(points_2d)

    # Filter triangles whose centroid is outside the disk (from Delaunay overshoot)
    filtered = []
    for tri in tris:
        cx = (points_2d[tri[0]][0] + points_2d[tri[1]][0] + points_2d[tri[2]][0]) / 3
        cy = (points_2d[tri[0]][1] + points_2d[tri[1]][1] + points_2d[tri[2]][1]) / 3
        if cx * cx + cy * cy <= R * R * 1.01:
            filtered.append(tri)
    tris = filtered

    # Write OBJ (vertices in xz plane, y=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# uniform disk, {len(points_2d)} verts, {len(tris)} faces\n")
        for (x, z) in points_2d:
            f.write(f"v {x} 0 {z}\n")
        for tri in tris:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"Wrote {len(points_2d)} vertices and {len(tris)} faces to {args.out}")


if __name__ == "__main__":
    main()
