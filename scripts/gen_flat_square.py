#!/usr/bin/env python3
"""Generate a flat triangulated square in the xz plane.

Output is a Wavefront OBJ with `v` and `f` lines only — matches what
src/graphics/meshloader.cpp parses. Used by the "Swelling of a Square"
checkpoint demo (paper Table 2 row 1).

Usage:
    python3 scripts/gen_flat_square.py [--cells N] [--side L] [--out PATH]
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=12,
                    help="grid cells per side (vertices = (N+1)^2, triangles = 2*N^2)")
    ap.add_argument("--side", type=float, default=1.0,
                    help="edge length of the square")
    ap.add_argument("--out", default="meshes/flat_square.obj",
                    help="output OBJ path")
    args = ap.parse_args()

    N = args.cells
    side = args.side

    verts = []
    for j in range(N + 1):
        for i in range(N + 1):
            x = -side / 2 + side * i / N
            z = -side / 2 + side * j / N
            verts.append((x, 0.0, z))

    faces = []
    for j in range(N):
        for i in range(N):
            a = j * (N + 1) + i
            b = j * (N + 1) + i + 1
            c = (j + 1) * (N + 1) + i + 1
            d = (j + 1) * (N + 1) + i
            # OBJ indices are 1-based.
            faces.append((a + 1, b + 1, c + 1))
            faces.append((a + 1, c + 1, d + 1))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# flat square in the xz plane, side={side}, {N} cells per side\n")
        for v in verts:
            f.write("v %f %f %f\n" % v)
        for face in faces:
            f.write("f %d %d %d\n" % face)

    print(f"wrote {len(verts)} vertices and {len(faces)} faces to {args.out}")


if __name__ == "__main__":
    main()
