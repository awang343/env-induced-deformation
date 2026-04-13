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

    // Conformal rest metric: g = (2/(1+r²))² · a0 per face centroid.
    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d c = (mesh.vertices[tri[0]] + mesh.vertices[tri[1]] + mesh.vertices[tri[2]]) / 3.0;
        double r2 = c.x() * c.x() + c.z() * c.z();
        double s  = 2.0 / (1.0 + r2);
        rest.aBar[f]    = (s * s) * a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // Small random perturbation to break the flat symmetry (paper §7.1).
    // The flat configuration is an unstable equilibrium; Newton's method
    // needs a nudge to find the buckled (spherical) energy minimum.
    for (int i = 0; i < nV; ++i) {
        double r2 = mesh.vertices[i].x() * mesh.vertices[i].x()
                   + mesh.vertices[i].z() * mesh.vertices[i].z();
        mesh.vertices[i].y() += 0.001 * (1.0 - r2);
    }
}
