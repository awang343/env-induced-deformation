#!/usr/bin/env python3
"""Generate a flat triangulated disk in the xz plane (y=0).

Used by the "Disk to Sphere" demo — stereographic rest metric on a flat
disk that relaxes into a hemisphere (paper Figure 3, Section 7.1).

Usage:
    python3 scripts/gen_flat_disk.py [--rings N] [--sectors N] [--radius R] [--out PATH]
"""

import argparse
import math
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rings", type=int, default=6,
                    help="number of concentric rings (default 6)")
    ap.add_argument("--sectors", type=int, default=12,
                    help="vertices per ring (default 12)")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="disk radius (default 1.0)")
    ap.add_argument("--out", default="meshes/flat_disk.obj",
                    help="output OBJ path")
    args = ap.parse_args()

    N_rings = args.rings
    N_sectors = args.sectors
    R = args.radius

    verts = [(0.0, 0.0, 0.0)]  # center vertex, index 0

    for k in range(1, N_rings + 1):
        r = R * k / N_rings
        for s in range(N_sectors):
            angle = 2.0 * math.pi * s / N_sectors
            verts.append((r * math.cos(angle), 0.0, r * math.sin(angle)))

    faces = []
    # Fan from center to ring 1.
    for s in range(N_sectors):
        i0 = 0
        i1 = 1 + s
        i2 = 1 + (s + 1) % N_sectors
        faces.append((i0 + 1, i1 + 1, i2 + 1))  # 1-indexed

    # Strips between ring k and ring k+1.
    for k in range(1, N_rings):
        base_inner = 1 + (k - 1) * N_sectors
        base_outer = 1 + k * N_sectors
        for s in range(N_sectors):
            i0 = base_inner + s
            i1 = base_outer + s
            i2 = base_outer + (s + 1) % N_sectors
            i3 = base_inner + (s + 1) % N_sectors
            faces.append((i0 + 1, i1 + 1, i2 + 1))
            faces.append((i0 + 1, i2 + 1, i3 + 1))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# flat disk in xz plane, radius=%g, %d rings, %d sectors\n"
                % (R, N_rings, N_sectors))
        for v in verts:
            f.write("v %f %f %f\n" % v)
        for face in faces:
            f.write("f %d %d %d\n" % face)

    print("wrote %d vertices and %d faces to %s" % (len(verts), len(faces), args.out))


if __name__ == "__main__":
    main()
