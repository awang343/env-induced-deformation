#!/usr/bin/env python3
"""Generate a sphere mesh as OBJ. Supports regular UV and random triangulations."""
import argparse
import math
import numpy as np
from scipy.spatial import ConvexHull


def uv_sphere(radius, rings, segments):
    verts = [(0, radius, 0)]
    for i in range(1, rings):
        phi = math.pi * i / rings
        y = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            verts.append((r * math.cos(theta), y, r * math.sin(theta)))
    verts.append((0, -radius, 0))

    faces = []
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((1, j + 2, j1 + 2))
    for i in range(rings - 2):
        for j in range(segments):
            j1 = (j + 1) % segments
            a = 1 + i * segments + j + 1
            b = 1 + i * segments + j1 + 1
            c = 1 + (i + 1) * segments + j + 1
            d = 1 + (i + 1) * segments + j1 + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    bot = len(verts)
    base = 1 + (rings - 2) * segments
    for j in range(segments):
        j1 = (j + 1) % segments
        faces.append((base + j + 1, bot, base + j1 + 1))
    return verts, faces


def random_sphere(radius, npts, seed):
    rng = np.random.default_rng(seed)

    # Fibonacci spiral for even distribution, then jitter.
    golden = (1 + np.sqrt(5)) / 2
    indices = np.arange(npts)
    theta = 2 * np.pi * indices / golden
    phi = np.arccos(1 - 2 * (indices + 0.5) / npts)

    # Add jitter proportional to average spacing.
    avg_spacing = np.sqrt(4 * np.pi / npts)
    theta += rng.uniform(-avg_spacing * 0.3, avg_spacing * 0.3, npts)
    phi += rng.uniform(-avg_spacing * 0.15, avg_spacing * 0.15, npts)
    phi = np.clip(phi, 0.01, np.pi - 0.01)

    pts = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.cos(phi),
        radius * np.sin(phi) * np.sin(theta),
    ])

    # Project back to sphere (jitter may have moved them off).
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * radius

    # Convex hull gives Delaunay triangulation on the sphere.
    hull = ConvexHull(pts)

    verts = [(p[0], p[1], p[2]) for p in pts]
    faces = [(s[0] + 1, s[1] + 1, s[2] + 1) for s in hull.simplices]
    return verts, faces


def main():
    p = argparse.ArgumentParser(description="Generate a sphere OBJ")
    p.add_argument("-o", "--output", default="meshes/sphere.obj")
    p.add_argument("-r", "--radius", type=float, default=1.0)
    p.add_argument("--rings", type=int, default=16, help="latitude divisions (UV mode)")
    p.add_argument("--segments", type=int, default=32, help="longitude divisions (UV mode)")
    p.add_argument("--random", action="store_true", help="random triangulation")
    p.add_argument("--npts", type=int, default=500, help="point count (random mode)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.random:
        verts, faces = random_sphere(args.radius, args.npts, args.seed)
    else:
        verts, faces = uv_sphere(args.radius, args.rings, args.segments)

    with open(args.output, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"Wrote {len(verts)} verts, {len(faces)} faces to {args.output}")


if __name__ == "__main__":
    main()
