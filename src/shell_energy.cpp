#include "shell_energy.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Eigen;

// =============================================================================
// Helpers
// =============================================================================

static Matrix3d skew(const Vector3d &v)
{
    Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

static double svNormSq(const Matrix2d &M, double alpha, double beta)
{
    double tr  = M.trace();
    double tr2 = (M * M).trace();
    return 0.5 * alpha * tr * tr + beta * tr2;
}

// =============================================================================
// Face normals (unnormalized)
// =============================================================================

void computeFaceNormals(const ShellMesh &mesh, std::vector<Vector3d> &normals)
{
    const int nF = mesh.numFaces();
    normals.resize(nF);
    for (int f = 0; f < nF; ++f) {
        const auto &t = mesh.faces[f];
        normals[f] = (mesh.vertices[t[1]] - mesh.vertices[t[0]])
                     .cross(mesh.vertices[t[2]] - mesh.vertices[t[0]]);
    }
}

// =============================================================================
// First fundamental form + derivatives
// =============================================================================

Matrix2d firstFundamentalForm(const ShellMesh &mesh, int face)
{
    const Vector3i &tri = mesh.faces[face];
    Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
    Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];
    double d12 = e1.dot(e2);
    Matrix2d a;
    a << e1.dot(e1), d12,
         d12,        e2.dot(e2);
    return a;
}

// Derivative of a w.r.t. 9 DOFs (v0,v1,v2 × xyz). Returns 4×9.
// Column-major ordering of a: [a(0,0), a(1,0), a(0,1), a(1,1)].
static Matrix<double, 4, 9> firstFFDeriv(const ShellMesh &mesh, int face)
{
    const Vector3i &tri = mesh.faces[face];
    Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
    Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];

    Matrix<double, 4, 9> D = Matrix<double, 4, 9>::Zero();
    // a(0,0) = e1·e1 → row 0
    D.block<1,3>(0, 0) = -2.0 * e1.transpose();
    D.block<1,3>(0, 3) =  2.0 * e1.transpose();
    // a(1,0) = e1·e2 → row 1
    D.block<1,3>(1, 0) = -(e1 + e2).transpose();
    D.block<1,3>(1, 3) =  e2.transpose();
    D.block<1,3>(1, 6) =  e1.transpose();
    // a(0,1) = e1·e2 → row 2 (same as row 1)
    D.row(2) = D.row(1);
    // a(1,1) = e2·e2 → row 3
    D.block<1,3>(3, 0) = -2.0 * e2.transpose();
    D.block<1,3>(3, 6) =  2.0 * e2.transpose();
    return D;
}

// Hessian of each entry of a. Returns 4 constant 9×9 matrices.
// These don't depend on vertex positions — only on the identity matrix.
static void firstFFHessian(Matrix<double, 9, 9> ahess[4])
{
    for (int i = 0; i < 4; ++i) ahess[i].setZero();
    Matrix3d I3 = Matrix3d::Identity();

    // a(0,0) = |e1|²: d²/dv0² = 2I, d²/dv1² = 2I, d²/dv0v1 = -2I
    ahess[0].block<3,3>(0, 0) =  2.0 * I3;
    ahess[0].block<3,3>(3, 3) =  2.0 * I3;
    ahess[0].block<3,3>(0, 3) = -2.0 * I3;
    ahess[0].block<3,3>(3, 0) = -2.0 * I3;

    // a(1,0) = a(0,1) = e1·e2
    ahess[1].block<3,3>(0, 0) =  2.0 * I3;
    ahess[1].block<3,3>(0, 3) = -I3;
    ahess[1].block<3,3>(0, 6) = -I3;
    ahess[1].block<3,3>(3, 0) = -I3;
    ahess[1].block<3,3>(3, 6) =  I3;
    ahess[1].block<3,3>(6, 0) = -I3;
    ahess[1].block<3,3>(6, 3) =  I3;
    ahess[2] = ahess[1]; // a(0,1) same

    // a(1,1) = |e2|²: d²/dv0² = 2I, d²/dv2² = 2I, d²/dv0v2 = -2I
    ahess[3].block<3,3>(0, 0) =  2.0 * I3;
    ahess[3].block<3,3>(6, 6) =  2.0 * I3;
    ahess[3].block<3,3>(0, 6) = -2.0 * I3;
    ahess[3].block<3,3>(6, 0) = -2.0 * I3;
}

// =============================================================================
// Second fundamental form (libshell MidedgeAverage) + derivative
// =============================================================================

Matrix2d secondFundamentalForm(const ShellMesh &mesh,
                               const std::vector<Vector3d> &fN,
                               int face)
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d n_center = fN[face];

    double II[3];
    for (int i = 0; i < 3; ++i) {
        int ip1 = (i + 1) % 3, ip2 = (i + 2) % 3;
        Vector3d qvec = q[tri[ip1]] + q[tri[ip2]] - 2.0 * q[tri[i]];
        int eid = mesh.faceEdges[face][i];
        const auto &refs = mesh.edgeFaces[eid];
        int oppFace = (refs[0].face == face) ? refs[1].face : refs[0].face;

        if (oppFace == -1) {
            II[i] = 0.0;
        } else {
            Vector3d n_opp = fN[oppFace];
            Vector3d mvec = n_opp + n_center;
            double mnorm = mvec.norm();
            II[i] = (mnorm > 0.0) ? qvec.dot(n_opp) / mnorm : 0.0;
        }
    }
    Matrix2d b;
    b << II[0] + II[1], II[0],
         II[0],         II[0] + II[2];
    return b;
}

// Derivative of b w.r.t. 18 DOFs. Returns 4×18 Jacobian.
// DOFs: [tri[0] xyz, tri[1] xyz, tri[2] xyz, opp0 xyz, opp1 xyz, opp2 xyz]
// where opp_i is the vertex opposite edge i in the neighbor face (-1 if boundary).
// Column-major b: [b(0,0), b(1,0), b(0,1), b(1,1)].
static Matrix<double, 4, 18> secondFFDeriv(
    const ShellMesh &mesh,
    const std::vector<Vector3d> &fN,
    int face,
    int oppVerts[3])   // outputs: global vertex index of each opposite vertex (-1 if boundary)
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d n_center = fN[face];
    Vector3d e1 = q[tri[1]] - q[tri[0]];
    Vector3d e2 = q[tri[2]] - q[tri[0]];

    // Center normal derivative: dn_center/d(v0,v1,v2) is 3×9
    // n_center = e1 × e2 = (v1-v0) × (v2-v0)
    Matrix<double, 3, 9> dn_center = Matrix<double, 3, 9>::Zero();
    dn_center.block<3,3>(0, 0) =  skew(q[tri[2]] - q[tri[1]]);  // dv0
    dn_center.block<3,3>(0, 3) = -skew(e2);                       // dv1
    dn_center.block<3,3>(0, 6) =  skew(e1);                       // dv2

    // Compute dII[i]/dDOF for each edge i.
    Matrix<double, 1, 18> dII[3];
    for (int i = 0; i < 3; ++i) dII[i].setZero();

    for (int i = 0; i < 3; ++i) {
        int ip1 = (i + 1) % 3, ip2 = (i + 2) % 3;
        Vector3d qvec = q[tri[ip1]] + q[tri[ip2]] - 2.0 * q[tri[i]];

        int eid = mesh.faceEdges[face][i];
        const auto &refs = mesh.edgeFaces[eid];
        int oppFace = (refs[0].face == face) ? refs[1].face : refs[0].face;
        int oppLocalVtx = (refs[0].face == face) ? refs[1].localOppVtx : refs[0].localOppVtx;

        if (oppFace == -1) {
            oppVerts[i] = -1;
            continue; // II[i] = 0, dII[i] = 0
        }

        // Find the opposite vertex (the one not on the shared edge).
        oppVerts[i] = mesh.faces[oppFace][oppLocalVtx];

        Vector3d n_opp = fN[oppFace];
        Vector3d mvec = n_opp + n_center;
        double mnorm = mvec.norm();
        if (mnorm < 1e-16) continue;

        double IIval = qvec.dot(n_opp) / mnorm;

        // ---- Part 1: from qvec (vertex positions in numerator) ----
        // dqvec/dv_i = -2I, dqvec/dv_{i+1} = +I, dqvec/dv_{i+2} = +I
        Vector3d n_opp_over_mn = n_opp / mnorm;
        dII[i].segment<3>(3 * i)   += -2.0 * n_opp_over_mn.transpose();
        dII[i].segment<3>(3 * ip1) +=        n_opp_over_mn.transpose();
        dII[i].segment<3>(3 * ip2) +=        n_opp_over_mn.transpose();

        // ---- Opposite face normal derivative (3×9 in opp face local DOFs) ----
        // n_opp = (opp_v_{k+1} - opp_v_k) × (opp_v_{k+2} - opp_v_k)
        // where k = oppLocalVtx.
        // The 3 vertices of oppFace, rotated so opp vertex comes first:
        int ov0 = mesh.faces[oppFace][oppLocalVtx];                      // opposite vertex
        int ov1 = mesh.faces[oppFace][(oppLocalVtx + 1) % 3];           // shared
        int ov2 = mesh.faces[oppFace][(oppLocalVtx + 2) % 3];           // shared
        Vector3d oe1 = q[ov1] - q[ov0];
        Vector3d oe2 = q[ov2] - q[ov0];

        // dn_opp / d(ov0, ov1, ov2):
        Matrix3d dn_opp_ov0 =  skew(q[ov2] - q[ov1]);
        Matrix3d dn_opp_ov1 = -skew(oe2);
        Matrix3d dn_opp_ov2 =  skew(oe1);

        // Map opp face vertices to 18-DOF indices:
        // ov0 = opposite vertex → DOF 9 + 3*i
        // ov1, ov2 = shared vertices = tri[(i+1)%3], tri[(i+2)%3]
        // But ov1/ov2 might map to either of the shared face vertices.
        // Find which face-local index each shared vertex corresponds to.
        auto findLocal = [&](int globalVtx) -> int {
            for (int j = 0; j < 3; ++j)
                if (tri[j] == globalVtx) return j;
            return -1;
        };
        int loc_ov1 = findLocal(ov1);
        int loc_ov2 = findLocal(ov2);

        // ---- Part 2: from n_opp in numerator (qvec · n_opp) ----
        Vector3d qvec_over_mn = qvec / mnorm;
        // Contribution from ov0 (opposite vertex, DOF 9+3*i):
        dII[i].segment<3>(9 + 3 * i) += qvec_over_mn.transpose() * dn_opp_ov0;
        // Contribution from ov1 (shared, DOF 3*loc_ov1):
        if (loc_ov1 >= 0)
            dII[i].segment<3>(3 * loc_ov1) += qvec_over_mn.transpose() * dn_opp_ov1;
        // Contribution from ov2 (shared, DOF 3*loc_ov2):
        if (loc_ov2 >= 0)
            dII[i].segment<3>(3 * loc_ov2) += qvec_over_mn.transpose() * dn_opp_ov2;

        // ---- Part 3: from ||mvec|| in denominator ----
        double coef = -IIval / (mnorm * mnorm);
        RowVector3d mvec_t = mvec.transpose();

        // From n_opp:
        dII[i].segment<3>(9 + 3 * i) += coef * mvec_t * dn_opp_ov0;
        if (loc_ov1 >= 0)
            dII[i].segment<3>(3 * loc_ov1) += coef * mvec_t * dn_opp_ov1;
        if (loc_ov2 >= 0)
            dII[i].segment<3>(3 * loc_ov2) += coef * mvec_t * dn_opp_ov2;

        // From n_center (affects face DOFs 0-8):
        for (int j = 0; j < 3; ++j)
            dII[i].segment<3>(3 * j) += coef * mvec_t * dn_center.block<3,3>(0, 3*j);
    }

    // Assemble 4×18 b derivative from dII[0..2].
    // b = [[II0+II1, II0], [II0, II0+II2]]
    // Column-major: b(0,0)=II0+II1, b(1,0)=II0, b(0,1)=II0, b(1,1)=II0+II2
    Matrix<double, 4, 18> bDeriv;
    bDeriv.row(0) = dII[0] + dII[1];       // b(0,0)
    bDeriv.row(1) = dII[0];                // b(1,0)
    bDeriv.row(2) = dII[0];                // b(0,1)
    bDeriv.row(3) = dII[0] + dII[2];       // b(1,1)
    return bDeriv;
}

// =============================================================================
// Per-face stretching: energy + gradient + Hessian
// =============================================================================

StretchingData stretchingPerFace(const ShellMesh &mesh,
                                 const ShellRestState &rest,
                                 const MaterialParams &mat,
                                 int face)
{
    StretchingData result;
    const double alpha = mat.alpha();
    const double beta  = mat.beta();

    Matrix2d a      = firstFundamentalForm(mesh, face);
    Matrix2d aBarInv = rest.aBar[face].inverse();
    Matrix2d M       = aBarInv * a - Matrix2d::Identity();
    double   coef    = 0.25 * mat.thickness * rest.restArea[face];

    // Energy
    result.energy = coef * svNormSq(M, alpha, beta);

    // Stress: dStVK/da (as 4-vector, column-major)
    Matrix2d stress_mat = alpha * M.trace() * aBarInv + 2.0 * beta * M * aBarInv;
    Map<Vector4d> stress_vec(stress_mat.data());

    // Gradient: coef * (da/dDOF)^T * stress_vec
    Matrix<double, 4, 9> aDeriv = firstFFDeriv(mesh, face);
    result.gradient = coef * aDeriv.transpose() * stress_vec;

    // Hessian (3 terms from libshell StVKMaterial)
    Matrix<double, 9, 9> ahess[4];
    firstFFHessian(ahess);

    // Term 1: α * (da^T · āInv_vec) (da^T · āInv_vec)^T
    Map<Vector4d> abarinv_vec(const_cast<double*>(aBarInv.data()));
    Matrix<double, 1, 9> inner = aDeriv.transpose() * abarinv_vec;
    // Actually inner should be 9×1 → let me fix
    Matrix<double, 9, 1> inner1 = aDeriv.transpose() * abarinv_vec;
    result.hessian = coef * alpha * inner1 * inner1.transpose();

    // Term 2: stress contracted with d²a
    Matrix2d Mainv = M * aBarInv;
    for (int k = 0; k < 4; ++k) {
        double s = alpha * M.trace() * aBarInv.data()[k] + 2.0 * beta * Mainv.data()[k];
        result.hessian += coef * s * ahess[k];
    }

    // Term 3: 2β * products of (āInv · da) rows
    // inner_IJ = āInv(I,0)*aDeriv.row(col_of_a(I,J)_first) + āInv(I,1)*aDeriv.row(col_of_a(I,J)_second)
    // For 2×2 column-major: col 0 = rows 0,1; col 1 = rows 2,3
    Matrix<double, 1, 9> inner00 = aBarInv(0,0) * aDeriv.row(0) + aBarInv(0,1) * aDeriv.row(1);
    Matrix<double, 1, 9> inner01 = aBarInv(0,0) * aDeriv.row(2) + aBarInv(0,1) * aDeriv.row(3);
    Matrix<double, 1, 9> inner10 = aBarInv(1,0) * aDeriv.row(0) + aBarInv(1,1) * aDeriv.row(1);
    Matrix<double, 1, 9> inner11 = aBarInv(1,0) * aDeriv.row(2) + aBarInv(1,1) * aDeriv.row(3);

    result.hessian += coef * 2.0 * beta * (
        inner00.transpose() * inner00 +
        inner01.transpose() * inner10 +
        inner10.transpose() * inner01 +
        inner11.transpose() * inner11);

    return result;
}

// =============================================================================
// Per-face bending: energy + gradient + inexact Hessian
// =============================================================================

BendingData bendingPerFace(const ShellMesh &mesh,
                           const ShellRestState &rest,
                           const MaterialParams &mat,
                           const std::vector<Vector3d> &fN,
                           int face)
{
    BendingData result;
    const double alpha = mat.alpha();
    const double beta  = mat.beta();
    const double h3_12 = mat.thickness * mat.thickness * mat.thickness / 12.0;
    const double coef  = h3_12 * rest.restArea[face];

    int oppVerts[3];
    Matrix<double, 4, 18> bDeriv = secondFFDeriv(mesh, fN, face, oppVerts);
    for (int i = 0; i < 3; ++i) result.vertIdx[i] = mesh.faces[face][i];
    for (int i = 0; i < 3; ++i) result.vertIdx[3 + i] = oppVerts[i];

    Matrix2d b       = secondFundamentalForm(mesh, fN, face);
    Matrix2d aBarInv = rest.aBar[face].inverse();
    Matrix2d Mb      = aBarInv * (b - rest.bBar[face]);

    // Energy
    result.energy = coef * svNormSq(Mb, alpha, beta);

    // Stress: dStVK/db (4-vector, column-major)
    Matrix2d stress_mat = alpha * Mb.trace() * aBarInv + 2.0 * beta * Mb * aBarInv;
    Map<Vector4d> stress_vec(stress_mat.data());

    // Gradient: coef * (db/dDOF)^T * stress_vec
    result.gradient = coef * bDeriv.transpose() * stress_vec;

    // Inexact Hessian (Gauss-Newton, paper Section 5):
    //   E = coef * (r1² + r2²)
    //   r1 = sqrt(α/2) * tr(Mb),  r2 = sqrt(β * tr(Mb²))
    //   H ≈ coef * (∇r1 ∇r1^T + ∇r2 ∇r2^T)
    //
    // ∇r1 = sqrt(α/2) * d(tr(Mb))/dDOF = sqrt(α/2) * bDeriv^T * āInv_vec
    // ∇r2 = sqrt(β) * d(sqrt(tr(Mb²)))/dDOF = sqrt(β)/(2*r2) * d(tr(Mb²))/dDOF
    //      where d(tr(Mb²))/dDOF = 2 * bDeriv^T * vec(Mb * āInv)

    Map<Vector4d> abarinv_vec(const_cast<double*>(aBarInv.data()));
    Matrix<double, 18, 1> dr1 = std::sqrt(0.5 * alpha) * bDeriv.transpose() * abarinv_vec;

    double trMb2 = (Mb * Mb).trace();
    Matrix2d MbAinv = Mb * aBarInv;
    Map<Vector4d> mbainv_vec(MbAinv.data());
    Matrix<double, 18, 1> dr2;
    if (trMb2 > 1e-16) {
        double r2val = std::sqrt(beta * trMb2);
        dr2 = (beta / r2val) * bDeriv.transpose() * mbainv_vec;
    } else {
        dr2.setZero();
    }

    result.hessian = coef * (dr1 * dr1.transpose() + dr2 * dr2.transpose());

    return result;
}

// =============================================================================
// Global energy
// =============================================================================

double totalEnergy(const ShellMesh &mesh,
                   const ShellRestState &rest,
                   const MaterialParams &mat)
{
    // Energy-only evaluation (no gradient/Hessian). Used by line search.
    const double alpha = mat.alpha();
    const double beta  = mat.beta();
    const double h3_12 = mat.thickness * mat.thickness * mat.thickness / 12.0;
    const int nF = mesh.numFaces();

    std::vector<Vector3d> fN;
    computeFaceNormals(mesh, fN);

    double total = 0.0;
    for (int f = 0; f < nF; ++f) {
        // Stretching
        Matrix2d a = firstFundamentalForm(mesh, f);
        Matrix2d Ms = rest.aBar[f].inverse() * a - Matrix2d::Identity();
        total += 0.25 * mat.thickness * rest.restArea[f] * svNormSq(Ms, alpha, beta);

        // Bending
        Matrix2d b  = secondFundamentalForm(mesh, fN, f);
        Matrix2d Mb = rest.aBar[f].inverse() * (b - rest.bBar[f]);
        total += h3_12 * rest.restArea[f] * svNormSq(Mb, alpha, beta);
    }
    return total;
}

// =============================================================================
// Global assembly: gradient + Hessian triplets
// =============================================================================

void assembleGradientAndHessian(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    const Eigen::Vector3d &gravity,
    const std::vector<double> &masses,
    VectorXd &grad,
    std::vector<Triplet<double>> &triplets)
{
    const int n  = mesh.numVerts();
    const int nF = mesh.numFaces();
    const int dim = 3 * n;

    grad.setZero(dim);
    triplets.clear();

    // Gravity contribution to gradient (note: grad = +∇E, force = -grad)
    for (int i = 0; i < n; ++i) {
        // Potential energy V_grav = -m*g·x → ∇V_grav = -m*g
        // We add this to the elastic gradient.
        // Actually, we want the TOTAL force = -∇E_elastic + m*g. The
        // Newton residual uses -force = ∇E_elastic - m*g. So we subtract
        // gravity from the gradient of elastic energy. We do this in the
        // Newton solver itself, not here.
    }

    std::vector<Vector3d> fN;
    computeFaceNormals(mesh, fN);

    for (int f = 0; f < nF; ++f) {
        // ---- Stretching ----
        auto sd = stretchingPerFace(mesh, rest, mat, f);
        const Vector3i &tri = mesh.faces[f];
        for (int i = 0; i < 3; ++i)
            grad.segment<3>(3 * tri[i]) += sd.gradient.segment<3>(3 * i);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        triplets.emplace_back(3*tri[i]+a, 3*tri[j]+b,
                                              sd.hessian(3*i+a, 3*j+b));

        // ---- Bending ----
        auto bd = bendingPerFace(mesh, rest, mat, fN, f);
        for (int i = 0; i < 6; ++i) {
            int vi = bd.vertIdx[i];
            if (vi < 0) continue;
            grad.segment<3>(3 * vi) += bd.gradient.segment<3>(3 * i);
            for (int j = 0; j < 6; ++j) {
                int vj = bd.vertIdx[j];
                if (vj < 0) continue;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        triplets.emplace_back(3*vi+a, 3*vj+b,
                                              bd.hessian(3*i+a, 3*j+b));
            }
        }
    }
}

// =============================================================================
// Implicit Euler with Newton
// =============================================================================

void stepImplicitEuler(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    const Vector3d &gravity,
    std::vector<double> &masses,
    std::vector<Eigen::Vector3d> &velocities,
    double dt,
    int maxIters,
    double tol)
{
    const int n = mesh.numVerts();
    const int dim = 3 * n;
    const double inv_dt2 = 1.0 / (dt * dt);

    // Inertial prediction: x̃ = x0 + dt*v0 + dt²*g
    auto pos0 = mesh.vertices;
    std::vector<Vector3d> xTilde(n);
    for (int i = 0; i < n; ++i)
        xTilde[i] = pos0[i] + dt * velocities[i] + dt * dt * gravity;

    // Initial guess = inertial prediction.
    mesh.vertices = xTilde;

    // Newton iteration to minimize the incremental potential:
    //   Φ(x) = ½/dt² ||x - x̃||²_M + E(x)
    // ∇Φ  = M(x-x̃)/dt² + ∇E
    // ∇²Φ = M/dt² + ∇²E
    //
    // When ∇²E is indefinite (StVK in compression), we regularize the
    // diagonal until SimplicialLDLT succeeds (paper Section 5).
    for (int iter = 0; iter < maxIters; ++iter) {
        VectorXd eGrad;
        std::vector<Triplet<double>> hTrip;
        assembleGradientAndHessian(mesh, rest, mat, gravity, masses, eGrad, hTrip);

        // ∇Φ
        VectorXd grad(dim);
        for (int i = 0; i < n; ++i) {
            Vector3d dx = mesh.vertices[i] - xTilde[i];
            grad.segment<3>(3*i) = masses[i] * inv_dt2 * dx
                                 + Vector3d(eGrad.segment<3>(3*i));
        }

        if (grad.norm() < tol) break;

        // ∇²Φ = M/dt² + ∇²E, with progressively stronger regularization
        // until the factorization succeeds.
        VectorXd dx;
        bool solved = false;
        for (double alpha = 0.0; alpha < 1e8; alpha = (alpha == 0.0) ? 1.0 : alpha * 2.0) {
            std::vector<Triplet<double>> sysTriplets;
            sysTriplets.reserve(hTrip.size() + dim);
            for (int i = 0; i < n; ++i) {
                double diag = (1.0 + alpha) * masses[i] * inv_dt2;
                for (int a = 0; a < 3; ++a)
                    sysTriplets.emplace_back(3*i+a, 3*i+a, diag);
            }
            for (auto &t : hTrip)
                sysTriplets.emplace_back(t.row(), t.col(), t.value());

            SparseMatrix<double> H(dim, dim);
            H.setFromTriplets(sysTriplets.begin(), sysTriplets.end());

            SimplicialLDLT<SparseMatrix<double>> solver;
            solver.compute(H);
            if (solver.info() == Eigen::Success) {
                dx = solver.solve(-grad);
                if (solver.info() == Eigen::Success) {
                    solved = true;
                    break;
                }
            }
        }
        if (!solved) break;

        // Take the full Newton step (no line search — the mass
        // regularization M/dt² ensures the step is bounded).
        for (int i = 0; i < n; ++i)
            mesh.vertices[i] += Vector3d(dx.segment<3>(3*i));
    }

    // Recover velocity: v = (x^{i+1} - x^i) / dt.
    for (int i = 0; i < n; ++i)
        velocities[i] = (mesh.vertices[i] - pos0[i]) / dt;
}

// =============================================================================
// Lumped mass
// =============================================================================

void computeLumpedMasses(const ShellMesh &mesh,
                         const ShellRestState &rest,
                         const MaterialParams &mat,
                         std::vector<double> &masses)
{
    masses.assign(mesh.numVerts(), 0.0);
    for (int f = 0; f < mesh.numFaces(); ++f) {
        double m = mat.density * mat.thickness * rest.restArea[f] / 3.0;
        for (int i = 0; i < 3; ++i) masses[mesh.faces[f][i]] += m;
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

void verifyForceGradient(ShellMesh &mesh,
                         const ShellRestState &rest,
                         const MaterialParams &mat,
                         const Vector3d &gravity,
                         const std::vector<double> &masses,
                         double eps)
{
    if (mesh.vertices.empty()) return;
    const int n = mesh.numVerts();
    auto saved = mesh.vertices;

    for (int i = 0; i < n; ++i)
        mesh.vertices[i] += Vector3d(
            0.01 * std::sin(0.7*i+1.0),
            0.01 * std::cos(1.3*i+0.4),
            0.01 * std::sin(0.5*i-0.2));

    // Analytical gradient from assembleGradientAndHessian
    VectorXd grad;
    std::vector<Triplet<double>> hTrip;
    assembleGradientAndHessian(mesh, rest, mat, gravity, masses, grad, hTrip);

    double worstAbs = 0.0, worstRel = 0.0;
    int worstIdx = -1, worstAxis = -1;

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            double orig = mesh.vertices[i][k];
            mesh.vertices[i][k] = orig + eps;
            double Ep = totalEnergy(mesh, rest, mat);
            mesh.vertices[i][k] = orig - eps;
            double Em = totalEnergy(mesh, rest, mat);
            mesh.vertices[i][k] = orig;

            double numF = (Ep - Em) / (2.0 * eps);  // +grad (not -force)
            double anaF = grad[3*i+k];
            double ae = std::abs(numF - anaF);
            double den = std::max({std::abs(numF), std::abs(anaF), 1e-12});
            double re = ae / den;
            if (ae > worstAbs) { worstAbs=ae; worstRel=re; worstIdx=i; worstAxis=k; }
        }
    }

    std::cout << "[force-FD check] worst |abs|=" << worstAbs
              << "  rel=" << worstRel
              << "  at vertex " << worstIdx << " axis " << worstAxis
              << "  (eps=" << eps << ")" << std::endl;

    mesh.vertices = saved;
}
