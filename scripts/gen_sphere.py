#!/usr/bin/env python3
"""Generate a UV sphere mesh as OBJ."""
import argparse, math, sys

def main():
    p = argparse.ArgumentParser(description="Generate a UV sphere OBJ")
    p.add_argument("-o", "--output", default="meshes/sphere.obj")
    p.add_argument("-r", "--radius", type=float, default=1.0)
    p.add_argument("--rings", type=int, default=16, help="latitude divisions")
    p.add_argument("--segments", type=int, default=32, help="longitude divisions")
    args = p.parse_args()

    verts = []
    faces = []
    R, rings, segs = args.radius, args.rings, args.segments

    # Top pole
    verts.append((0, R, 0))

    # Middle rings
    for i in range(1, rings):
        phi = math.pi * i / rings
        y = R * math.cos(phi)
        r = R * math.sin(phi)
        for j in range(segs):
            theta = 2 * math.pi * j / segs
            verts.append((r * math.cos(theta), y, r * math.sin(theta)))

    # Bottom pole
    verts.append((0, -R, 0))

    # Top cap
    for j in range(segs):
        j1 = (j + 1) % segs
        faces.append((1, j + 2, j1 + 2))

    # Middle quads (split into triangles)
    for i in range(rings - 2):
        for j in range(segs):
            j1 = (j + 1) % segs
            a = 1 + i * segs + j + 1
            b = 1 + i * segs + j1 + 1
            c = 1 + (i + 1) * segs + j + 1
            d = 1 + (i + 1) * segs + j1 + 1
            faces.append((a, c, b))
            faces.append((b, c, d))

    # Bottom cap
    bot = len(verts)
    base = 1 + (rings - 2) * segs
    for j in range(segs):
        j1 = (j + 1) % segs
        faces.append((base + j + 1, bot, base + j1 + 1))

    with open(args.output, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"Wrote {len(verts)} verts, {len(faces)} faces to {args.output}")

if __name__ == "__main__":
    main()
