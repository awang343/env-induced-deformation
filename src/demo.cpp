#include "demo.h"

#include <cmath>
#include <random>

using namespace Eigen;

// Shared: stereographic projection to unit sphere.
static Vector3d stereoProject(const Vector3d &v)
{
    double r2 = v.x() * v.x() + v.z() * v.z();
    double d  = 1.0 + r2;
    return Vector3d(2.0 * v.x() / d, (1.0 - r2) / d, 2.0 * v.z() / d);
}

void initSphereStretching(ShellMesh &mesh,
                          const std::vector<Matrix2d> &a0,
                          ShellRestState &rest,
                          int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    auto randPerturb = [&]() -> Vector3d {
        return perturbScale * Vector3d(dist(rng), dist(rng), dist(rng));
    };

    // āBar from sphere geometry.
    std::vector<Vector3d> sphereVerts(nV);
    for (int i = 0; i < nV; ++i)
        sphereVerts[i] = stereoProject(mesh.vertices[i]) + randPerturb();

    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d e1 = sphereVerts[tri[1]] - sphereVerts[tri[0]];
        Vector3d e2 = sphereVerts[tri[2]] - sphereVerts[tri[0]];
        double d12 = e1.dot(e2);
        rest.aBar[f] << e1.dot(e1), d12, d12, e2.dot(e2);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // b̄ = 0 (paper Figure 3 caption).

    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += randPerturb();
}

void initSphereBending(ShellMesh &mesh,
                       const std::vector<Matrix2d> &a0,
                       ShellRestState &rest,
                       int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    auto randPerturb = [&]() -> Vector3d {
        return perturbScale * Vector3d(dist(rng), dist(rng), dist(rng));
    };

    // āBar = flat disk (no stretching drive).
    for (int f = 0; f < nF; ++f) {
        rest.aBar[f] = a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, a0[f].determinant()));
    }

    // b̄ from sphere curvature.
    std::vector<Vector3d> sphereVerts(nV);
    for (int i = 0; i < nV; ++i)
        sphereVerts[i] = stereoProject(mesh.vertices[i]);

    auto flatVerts = mesh.vertices;
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] = sphereVerts[i];

    std::vector<Vector3d> sphereFN;
    computeFaceNormals(mesh, sphereFN);
    for (int f = 0; f < nF; ++f)
        rest.bBar[f] = secondFundamentalForm(mesh, sphereFN, f);

    mesh.vertices = flatVerts;

    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += randPerturb();
}

void initIsotropicGrowth(ShellMesh &mesh,
                         const std::vector<Matrix2d> &a0,
                         ShellRestState &rest,
                         double growthFactor,
                         int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    // āBar = s² · a₀ (rest edges are s× longer than current).
    const double s2 = growthFactor * growthFactor;
    for (int f = 0; f < nF; ++f) {
        rest.aBar[f] = s2 * a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // b̄ = b⁰ (initial curvature preserved, per paper Section 2).
    std::vector<Vector3d> fN;
    computeFaceNormals(mesh, fN);
    for (int f = 0; f < nF; ++f)
        rest.bBar[f] = secondFundamentalForm(mesh, fN, f);

    // Small perturbation to break symmetry (only matters for flat meshes).
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i].y() += perturbScale * dist(rng);
}

void initCylinderDemo(ShellMesh &mesh,
                      const std::vector<Matrix2d> &a0,
                      ShellRestState &rest,
                      double kappa,
                      int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    // āBar = flat metric (cylinder is developable).
    for (int f = 0; f < nF; ++f) {
        rest.aBar[f] = a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, a0[f].determinant()));
    }

    // b̄ for curvature κ=1 in the x-direction, expressed in each face's
    // local barycentric coordinates: b̄_ij = κ · (e_i · x̂)(e_j · x̂)
    // This ensures all faces agree on a consistent global curl direction.
    // kappa passed as parameter
    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
        Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];
        double e1x = e1.x(), e2x = e2.x();
        rest.bBar[f] << kappa*e1x*e1x, kappa*e1x*e2x,
                         kappa*e1x*e2x, kappa*e2x*e2x;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i].y() += perturbScale * dist(rng);
}
