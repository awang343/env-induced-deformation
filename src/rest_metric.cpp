#include "rest_metric.h"

#include <cmath>
#include <random>

using namespace Eigen;

// Average face normal of the mesh (for perturbation direction).
static Vector3d averageNormal(const ShellMesh &mesh)
{
    std::vector<Vector3d> fN;
    computeFaceNormals(mesh, fN);
    Vector3d avg = Vector3d::Zero();
    for (auto &n : fN) avg += n;
    double len = avg.norm();
    return (len > 0) ? Vector3d(avg / len) : Vector3d::UnitY();
}

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

    // Small perturbation along mesh normal to break symmetry.
    Vector3d avgN = averageNormal(mesh);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += perturbScale * dist(rng) * avgN;
}

void initSwelling(ShellMesh &mesh,
                  const std::vector<Matrix2d> &a0,
                  ShellRestState &rest,
                  int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    // Initialize rest forms to current geometry (no deformation yet).
    // āBar = a₀, b̄ = b₀. These will be updated each step from moisture.
    for (int f = 0; f < nF; ++f) {
        rest.aBar[f] = a0[f];
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, a0[f].determinant()));
    }

    std::vector<Vector3d> fN;
    computeFaceNormals(mesh, fN);
    for (int f = 0; f < nF; ++f)
        rest.bBar[f] = secondFundamentalForm(mesh, fN, f);

    // Small perturbation along mesh normal for symmetry breaking.
    Vector3d avgN = averageNormal(mesh);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += perturbScale * dist(rng) * avgN;
}

void initSphereTarget(ShellMesh &mesh,
                      const std::vector<Matrix2d> &a0,
                      ShellRestState &rest,
                      double radius, int seed, double perturbScale)
{
    const int nF = mesh.numFaces();
    const int nV = mesh.numVerts();

    // Center the mesh and project vertices onto unit sphere.
    Vector3d center = Vector3d::Zero();
    for (int i = 0; i < nV; ++i) center += mesh.vertices[i];
    center /= nV;

    std::vector<Vector3d> sphereVerts(nV);
    for (int i = 0; i < nV; ++i) {
        Vector3d v = mesh.vertices[i] - center;
        double len = v.norm();
        sphereVerts[i] = (len > 1e-12) ? Vector3d(radius * v / len) : Vector3d(radius * Vector3d::UnitY());
    }

    // ā from sphere geometry.
    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        Vector3d e1 = sphereVerts[tri[1]] - sphereVerts[tri[0]];
        Vector3d e2 = sphereVerts[tri[2]] - sphereVerts[tri[0]];
        double d12 = e1.dot(e2);
        rest.aBar[f] << e1.dot(e1), d12, d12, e2.dot(e2);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }

    // b̄ from sphere curvature.
    auto origVerts = mesh.vertices;
    mesh.vertices = sphereVerts;
    std::vector<Vector3d> sphereFN;
    computeFaceNormals(mesh, sphereFN);
    for (int f = 0; f < nF; ++f)
        rest.bBar[f] = secondFundamentalForm(mesh, sphereFN, f);
    mesh.vertices = origVerts;

    // Perturbation for symmetry breaking.
    Vector3d avgN = averageNormal(mesh);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += perturbScale * dist(rng) * avgN;
}

// =============================================================================
// Swelling rest form updates (paper Section 2, page 6)
// =============================================================================

void updateRestFormsLinear(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Matrix2d> &a0,
    const std::vector<Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    double h, double mu)
{
    const int nF = mesh.numFaces();
    for (int f = 0; f < nF; ++f) {
        const auto &tri = mesh.faces[f];
        double mp = (mPlus[tri[0]] + mPlus[tri[1]] + mPlus[tri[2]]) / 3.0;
        double mm = (mMinus[tri[0]] + mMinus[tri[1]] + mMinus[tri[2]]) / 3.0;
        double sp = (1.0 + mp * mu); sp *= sp;  // (1 + m⁺μ)²
        double sm = (1.0 + mm * mu); sm *= sm;  // (1 + m⁻μ)²

        Matrix2d gPlus  = sp * (a0[f] - h * b0[f]);
        Matrix2d gMinus = sm * (a0[f] + h * b0[f]);

        rest.aBar[f] = 0.5 * (gPlus + gMinus);
        rest.bBar[f] = (gMinus - gPlus) / (2.0 * h);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }
}

void updateRestFormsPiecewise(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Matrix2d> &a0,
    const std::vector<Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    double h, double mu)
{
    const int nF = mesh.numFaces();
    for (int f = 0; f < nF; ++f) {
        const auto &tri = mesh.faces[f];
        double mp = (mPlus[tri[0]] + mPlus[tri[1]] + mPlus[tri[2]]) / 3.0;
        double mm = (mMinus[tri[0]] + mMinus[tri[1]] + mMinus[tri[2]]) / 3.0;
        double sp = (1.0 + mp * mu); sp *= sp;
        double sm = (1.0 + mm * mu); sm *= sm;

        double twoThirdsH = (2.0 / 3.0) * h;
        Matrix2d gPlus  = sp * (a0[f] - twoThirdsH * b0[f]);
        Matrix2d gMinus = sm * (a0[f] + twoThirdsH * b0[f]);

        rest.aBar[f] = 0.5 * (gPlus + gMinus);
        rest.bBar[f] = (3.0 / (4.0 * h)) * (gMinus - gPlus);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }
}

void updateRestFormsMachine(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Matrix2d> &a0,
    const std::vector<Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    const std::vector<Vector2d> &machineDir,
    double h, double mu, double muPerp)
{
    const int nF = mesh.numFaces();
    for (int f = 0; f < nF; ++f) {
        const auto &tri = mesh.faces[f];
        double mp = (mPlus[tri[0]] + mPlus[tri[1]] + mPlus[tri[2]]) / 3.0;
        double mm = (mMinus[tri[0]] + mMinus[tri[1]] + mMinus[tri[2]]) / 3.0;

        // Machine direction d and perpendicular d⊥ in barycentric coords.
        Vector2d d = machineDir[f].normalized();
        Vector2d dperp(-d.y(), d.x());

        // T = [d, d⊥]⁻¹ transforms from barycentric to d,d⊥ system.
        Matrix2d Tcols;
        Tcols.col(0) = d;
        Tcols.col(1) = dperp;
        Matrix2d Tinv = Tcols.inverse();

        // M± = diag(1 + m±μ, 1 + m±μ⊥)
        Matrix2d Mplus = Matrix2d::Zero();
        Mplus(0,0) = 1.0 + mp * mu;
        Mplus(1,1) = 1.0 + mp * muPerp;

        Matrix2d Mminus = Matrix2d::Zero();
        Mminus(0,0) = 1.0 + mm * mu;
        Mminus(1,1) = 1.0 + mm * muPerp;

        // g± = Tᵀ M± T⁻ᵀ (a⁰ ∓ hb⁰) T⁻¹ M± T
        Matrix2d TtMp = Tcols.transpose() * Mplus;
        Matrix2d TiMpT = Tinv * Mplus * Tcols;
        Matrix2d gPlus  = TtMp * Tinv.transpose() * (a0[f] - h * b0[f]) * TiMpT;

        Matrix2d TtMm = Tcols.transpose() * Mminus;
        Matrix2d TiMmT = Tinv * Mminus * Tcols;
        Matrix2d gMinus = TtMm * Tinv.transpose() * (a0[f] + h * b0[f]) * TiMmT;

        rest.aBar[f] = 0.5 * (gPlus + gMinus);
        rest.bBar[f] = (gMinus - gPlus) / (2.0 * h);
        rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, rest.aBar[f].determinant()));
    }
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

    Vector3d avgN = averageNormal(mesh);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (int i = 0; i < nV; ++i)
        mesh.vertices[i] += perturbScale * dist(rng) * avgN;
}
