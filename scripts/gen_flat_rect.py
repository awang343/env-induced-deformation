#!/usr/bin/env python3
"""Generate a flat rectangular mesh in the xz plane.
Used for the cylinder curling demo.

Usage:
    python3 scripts/gen_flat_rect.py [--nx N] [--nz N] [--width W] [--height H] [--out PATH]
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=10, help="cells along x")
    ap.add_argument("--nz", type=int, default=20, help="cells along z")
    ap.add_argument("--width", type=float, default=1.0, help="width along x")
    ap.add_argument("--height", type=float, default=2.0, help="height along z")
    ap.add_argument("--plane", choices=["xy", "xz"], default="xy",
                    help="which plane the rectangle lies in (default xy)")
    ap.add_argument("--out", default="meshes/flat_rect.obj", help="output path")
    args = ap.parse_args()

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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# flat rectangle {args.width}x{args.height}, {args.nx}x{args.nz} cells\n")
        for v in verts:
            f.write("v %f %f %f\n" % v)
        for face in faces:
            f.write("f %d %d %d\n" % face)

    print(f"Wrote {len(verts)} vertices and {len(faces)} faces to {args.out}")


if __name__ == "__main__":
    main()
