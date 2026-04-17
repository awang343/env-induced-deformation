#include "demo.h"

#include <cmath>
#include <iostream>
#include <random>

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
    // Sphere of radius R = 1 so the hemisphere has the same diameter
    // as the flat disk (matching the paper's Figure 3 visuals).
    const double R = 1.0;
    auto stereoProject = [R](const Vector3d &v) -> Vector3d {
        double r2 = v.x() * v.x() + v.z() * v.z();
        double d  = 1.0 + r2;
        return Vector3d(2.0 * R * v.x() / d,
                        R * (1.0 - r2) / d,
                        2.0 * R * v.z() / d);
    };

    // Paper §7.1: "We apply a small random perturbation to the rest
    // and initial configurations of all of our initially-flat examples
    // with b̄ = 0, to force symmetry-breaking."
    //
    // Perturb both the rest configuration (sphere vertices used to
    // compute āBar) and the initial mesh positions.
    const double perturbScale = 0.05;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    auto randPerturbation = [&]() -> Vector3d {
        return perturbScale * Vector3d(dist(rng), dist(rng), dist(rng));
    };

    // Perturbed sphere vertices → rest metric āBar.
    std::vector<Vector3d> sphereVerts(nV);
    for (int i = 0; i < nV; ++i)
        sphereVerts[i] = stereoProject(mesh.vertices[i]) + randPerturbation();

    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d e1 = sphereVerts[tri[1]] - sphereVerts[tri[0]];
        Vector3d e2 = sphereVerts[tri[2]] - sphereVerts[tri[0]];
        double d12 = e1.dot(e2);
        rest.aBar[f] << e1.dot(e1), d12,
                         d12,        e2.dot(e2);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // b̄ = 0 per the paper (Figure 3 caption).
    // (rest.bBar is already zero from init.)

    // Perturb initial mesh positions.
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += randPerturbation();
}
