#!/usr/bin/env python3
"""Scale and center an OBJ mesh. Writes result to stdout or a file."""
import argparse, sys

def main():
    p = argparse.ArgumentParser(description="Scale and center an OBJ mesh")
    p.add_argument("input", help="Input OBJ file")
    p.add_argument("-o", "--output", help="Output file (default: stdout)")
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   help="Target half-extent (max coord will be this value)")
    p.add_argument("--no-center", action="store_true", help="Don't center the mesh")
    args = p.parse_args()

    verts = []
    lines = []
    with open(args.input) as f:
        for line in f:
            lines.append(line)
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not verts:
        print("No vertices found", file=sys.stderr)
        sys.exit(1)

    # Center
    if not args.no_center:
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        for v in verts:
            v[0] -= cx; v[1] -= cy; v[2] -= cz

    # Scale
    max_coord = max(abs(c) for v in verts for c in v)
    if max_coord > 0:
        s = args.scale / max_coord
        for v in verts:
            v[0] *= s; v[1] *= s; v[2] *= s

    # Write
    out = open(args.output, "w") if args.output else sys.stdout
    vi = 0
    for line in lines:
        if line.startswith("v "):
            v = verts[vi]
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            vi += 1
        else:
            out.write(line)
    if args.output:
        out.close()

if __name__ == "__main__":
    main()
