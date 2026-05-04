#!/usr/bin/env python3
"""Generate a flat rectangular mesh.

Usage:
    python3 scripts/gen_flat_rect.py [--nx N] [--nz N] [--width W] [--height H] [--out PATH]
    python3 scripts/gen_flat_rect.py --random [--npts N] [--width W] [--height H] [--out PATH]
"""

import argparse
import os
import numpy as np
from scipy.spatial import Delaunay


def grid_mesh(args):
    verts = []
    for j in range(args.nz + 1):
        for i in range(args.nx + 1):
            u = -args.width / 2 + args.width * i / args.nx
            v = -args.height / 2 + args.height * j / args.nz
            if args.plane == "xy":
                verts.append((u, v, 0.0))
            else:
                verts.append((u, 0.0, v))

    faces = []
    for j in range(args.nz):
        for i in range(args.nx):
            a = j * (args.nx + 1) + i
            b = a + 1
            c = a + args.nx + 2
            d = a + args.nx + 1
            faces.append((a + 1, b + 1, c + 1))
            faces.append((a + 1, c + 1, d + 1))
    return verts, faces


def random_mesh(args):
    rng = np.random.default_rng(args.seed)
    w, h = args.width, args.height

    # Interior points with jittered grid for decent spacing.
    n = args.npts
    cols = int(np.sqrt(n * w / h))
    rows = int(np.sqrt(n * h / w))
    cols = max(cols, 2)
    rows = max(rows, 2)
    dx, dy = w / cols, h / rows
    pts = []
    for j in range(rows):
        for i in range(cols):
            x = -w/2 + dx * (i + 0.5) + rng.uniform(-dx*0.35, dx*0.35)
            y = -h/2 + dy * (j + 0.5) + rng.uniform(-dy*0.35, dy*0.35)
            # Clamp to stay inside
            x = np.clip(x, -w/2 + 1e-6, w/2 - 1e-6)
            y = np.clip(y, -h/2 + 1e-6, h/2 - 1e-6)
            pts.append([x, y])

    # Boundary points for clean edges.
    edge_spacing = min(dx, dy) * 0.8
    for x in np.linspace(-w/2, w/2, max(3, int(w / edge_spacing) + 1)):
        pts.append([x, -h/2])
        pts.append([x,  h/2])
    for y in np.linspace(-h/2, h/2, max(3, int(h / edge_spacing) + 1)):
        pts.append([-w/2, y])
        pts.append([ w/2, y])

    pts = np.array(pts)

    # Remove near-duplicates.
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    to_remove = set()
    for i in range(len(pts)):
        if i in to_remove:
            continue
        neighbors = tree.query_ball_point(pts[i], edge_spacing * 0.3)
        for j in neighbors:
            if j > i:
                to_remove.add(j)
    mask = [i not in to_remove for i in range(len(pts))]
    pts = pts[mask]

    tri = Delaunay(pts)
    faces_out = []
    verts_out = []
    for p in pts:
        if args.plane == "xy":
            verts_out.append((p[0], p[1], 0.0))
        else:
            verts_out.append((p[0], 0.0, p[1]))

    for s in tri.simplices:
        faces_out.append((s[0] + 1, s[1] + 1, s[2] + 1))

    return verts_out, faces_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=10, help="cells along x (grid mode)")
    ap.add_argument("--nz", type=int, default=20, help="cells along z (grid mode)")
    ap.add_argument("--width", type=float, default=1.0, help="width along x")
    ap.add_argument("--height", type=float, default=2.0, help="height along z")
    ap.add_argument("--plane", choices=["xy", "xz"], default="xy",
                    help="which plane the rectangle lies in (default xy)")
    ap.add_argument("--random", action="store_true", help="use random triangulation")
    ap.add_argument("--npts", type=int, default=200, help="approx point count (random mode)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", default="meshes/flat_rect.obj", help="output path")
    args = ap.parse_args()

    if args.random:
        verts, faces = random_mesh(args)
    else:
        verts, faces = grid_mesh(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# flat rectangle {args.width}x{args.height}\n")
        for v in verts:
            f.write("v %f %f %f\n" % v)
        for face in faces:
            f.write("f %d %d %d\n" % face)

    print(f"Wrote {len(verts)} vertices and {len(faces)} faces to {args.out}")


if __name__ == "__main__":
    main()
