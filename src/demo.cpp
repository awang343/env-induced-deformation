#include "demo.h"

#include <cmath>
#include <iostream>

using namespace Eigen;

// =============================================================================
// Uniform growth
// =============================================================================

void applyGrowthFactor(double factor,
                       const std::vector<Matrix2d> &a0,
                       ShellRestState &rest)
{
    const double s2 = factor * factor;
    const int nF = static_cast<int>(a0.size());
    for (int f = 0; f < nF; ++f) {
        rest.aBar[f]    = s2 * a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }
}

bool stepGrowthRamp(GrowthState &gs, double dt,
                    const std::vector<Matrix2d> &a0,
                    ShellRestState &rest)
{
    if (std::abs(gs.factor - gs.target) < 1e-9) return false;
    double sign = (gs.target > gs.factor) ? 1.0 : -1.0;
    double next = gs.factor + sign * gs.rate * dt;
    if (sign > 0 && next > gs.target) next = gs.target;
    if (sign < 0 && next < gs.target) next = gs.target;
    gs.factor = next;
    applyGrowthFactor(next, a0, rest);
    return true;
}

void cycleGrowthDemo(GrowthState &gs, bool &paused)
{
    static const double step = 0.2;
    static const double maxTarget = 1.6;
    static const double minTarget = 1.0;
    static int direction = +1;

    double next = gs.target + direction * step;
    if (next > maxTarget + 1e-9) {
        direction = -1;
        next = gs.target + direction * step;
    } else if (next < minTarget - 1e-9) {
        direction = +1;
        next = gs.target + direction * step;
    }
    gs.target = next;
    if (paused) {
        paused = false;
        std::cout << "Auto-unpaused" << std::endl;
    }
    std::cout << "Growth target = " << gs.target << std::endl;
}

// =============================================================================
// Stereographic disk-to-sphere
// =============================================================================

void initStereographicDemo(ShellMesh &mesh,
                           const std::vector<Matrix2d> &a0,
                           ShellRestState &rest,
                           double initBlend)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    // Compute aBar from the discrete geometry: map each vertex onto the
    // unit sphere via inverse stereographic projection, then compute the
    // first fundamental form of each triangle from the mapped positions.
    // This respects the actual edge lengths on the sphere rather than
    // evaluating a continuous conformal factor at the centroid.
    auto stereoProject = [](const Vector3d &v) -> Vector3d {
        double r2 = v.x() * v.x() + v.z() * v.z();
        double d  = 1.0 + r2;
        return Vector3d(2.0 * v.x() / d,
                        (1.0 - r2) / d,
                        2.0 * v.z() / d);
    };

    std::vector<Vector3d> sphereVerts(nV);
    for (int i = 0; i < nV; ++i)
        sphereVerts[i] = stereoProject(mesh.vertices[i]);

    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d e1 = sphereVerts[tri[1]] - sphereVerts[tri[0]];
        Vector3d e2 = sphereVerts[tri[2]] - sphereVerts[tri[0]];
        double d12 = e1.dot(e2);
        rest.aBar[f] << e1.dot(e1), d12,
                         d12,        e2.dot(e2);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // bBar = 0 per the paper (Figure 3 caption): "we set b̄ = 0 and ā
    // to this conformal metric." The stretching energy alone drives the
    // shape; bending with b̄ = 0 provides smoothness. For thin shells
    // (h << L), stretching dominates by 1/h² and the equilibrium is
    // very close to the sphere.
    // (rest.bBar is already zero from init.)

    // Small perturbation to break the flat symmetry (paper §7.1).
    // The flat configuration is an unstable equilibrium; Newton's method
    // needs a nudge to find the buckled (spherical) energy minimum.
    for (int i = 0; i < nV; ++i) {
        double r2 = mesh.vertices[i].x() * mesh.vertices[i].x()
                   + mesh.vertices[i].z() * mesh.vertices[i].z();
        mesh.vertices[i].y() += 0.001 * (1.0 - r2);
    }
}
