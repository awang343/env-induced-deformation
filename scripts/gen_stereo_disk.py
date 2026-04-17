#!/usr/bin/env python3
"""Generate a flat disk by sampling uniformly on the unit sphere's upper
hemisphere, then projecting to the plane via forward stereographic projection.

Points near the equator (disk boundary) are denser because the sphere has
more surface area there. This matches the stereographic rest metric's
resolution needs — the conformal factor varies most near the boundary.

Usage:
    python3 scripts/gen_stereo_disk.py [--target-verts N] [--out PATH]
"""

import argparse
import math
import os


def fibonacci_hemisphere(n_total):
    """Fibonacci spiral sampling on the upper hemisphere (y >= 0).
    Returns approximately n_total points."""
    # Generate ~2*n_total on the full sphere, keep upper hemisphere.
    n_sphere = n_total * 2
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for i in range(n_sphere):
        # Fibonacci sphere: theta from arccos formula, phi from golden angle
        y = 1.0 - (2.0 * i + 1.0) / n_sphere  # y in [-1, 1]
        if y < 0:
            continue
        r = math.sqrt(1.0 - y * y)
        phi = 2.0 * math.pi * i / golden
        x = r * math.cos(phi)
        z = r * math.sin(phi)
        points.append((x, y, z))
    return points


def stereo_forward(x, y, z):
    """Forward stereographic projection: sphere (x,y,z) → plane (u,w).
    Projects from south pole; north pole (0,1,0) → origin."""
    denom = 1.0 + y
    if denom < 1e-12:
        return None  # south pole maps to infinity
    return (x / denom, z / denom)


def delaunay_triangulate(points):
    """Bowyer-Watson Delaunay triangulation."""
    big = 10.0
    st = [(-big, -big), (big, -big), (0, big)]
    n_st = len(points)
    all_pts = list(points) + st
    triangles = [(n_st, n_st + 1, n_st + 2)]

    def in_circumcircle(tri, px, py):
        ax, ay = all_pts[tri[0]]
        bx, by = all_pts[tri[1]]
        cx, cy = all_pts[tri[2]]
        dx = ax - px; dy = ay - py
        ex = bx - px; ey = by - py
        fx = cx - px; fy = cy - py
        det = (dx * (ey * (fx*fx + fy*fy) - fy * (ex*ex + ey*ey))
             - dy * (ex * (fx*fx + fy*fy) - fx * (ex*ex + ey*ey))
             + (dx*dx + dy*dy) * (ex * fy - ey * fx))
        orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return det > 0 if orient > 0 else det < 0

    for i in range(len(points)):
        px, py = all_pts[i]
        bad = [t for t in triangles if in_circumcircle(t, px, py)]
        edges = []
        for tri in bad:
            for j in range(3):
                e = (tri[j], tri[(j + 1) % 3])
                shared = any(
                    other != tri and e[0] in other and e[1] in other
                    for other in bad
                )
                if not shared:
                    edges.append(e)
        for tri in bad:
            triangles.remove(tri)
        for e in edges:
            triangles.append((i, e[0], e[1]))

    return [t for t in triangles if all(v < n_st for v in t)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-verts", type=int, default=500,
                    help="approximate vertex count (default 500)")
    ap.add_argument("--out", default="meshes/flat_disk.obj",
                    help="output OBJ path")
    args = ap.parse_args()

    # Sample on hemisphere and project to disk.
    sphere_pts = fibonacci_hemisphere(args.target_verts)
    disk_pts = []
    for (x, y, z) in sphere_pts:
        proj = stereo_forward(x, y, z)
        if proj and (proj[0]**2 + proj[1]**2) <= 1.01:
            disk_pts.append(proj)

    # Ensure center point exists.
    has_center = any(u*u + w*w < 0.001 for (u, w) in disk_pts)
    if not has_center:
        disk_pts.insert(0, (0.0, 0.0))

    # Add explicit boundary points for clean edge.
    n_boundary = max(24, int(2 * math.pi / 0.05))
    for i in range(n_boundary):
        angle = 2 * math.pi * i / n_boundary
        bx = math.cos(angle)
        bz = math.sin(angle)
        too_close = any((bx-u)**2 + (bz-w)**2 < 0.001 for (u,w) in disk_pts)
        if not too_close:
            disk_pts.append((bx, bz))

    print(f"Got {len(disk_pts)} vertices, triangulating...")
    tris = delaunay_triangulate(disk_pts)

    # Filter triangles outside disk.
    filtered = []
    for tri in tris:
        cx = (disk_pts[tri[0]][0] + disk_pts[tri[1]][0] + disk_pts[tri[2]][0]) / 3
        cy = (disk_pts[tri[0]][1] + disk_pts[tri[1]][1] + disk_pts[tri[2]][1]) / 3
        if cx*cx + cy*cy <= 1.01:
            filtered.append(tri)
    tris = filtered

    # Write OBJ (vertices in xz plane, y=0).
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# stereo-sampled disk, {len(disk_pts)} verts, {len(tris)} faces\n")
        for (u, w) in disk_pts:
            f.write(f"v {u} 0 {w}\n")
        for tri in tris:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"Wrote {len(disk_pts)} vertices and {len(tris)} faces to {args.out}")


if __name__ == "__main__":
    main()
