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

// Project a symmetric matrix to positive semi-definite by clamping
// negative eigenvalues to zero (libshell's projSymMatrix with kMaxZero).
template<int N>
static Matrix<double,N,N> projectPSD(const Matrix<double,N,N> &H)
{
    SelfAdjointEigenSolver<Matrix<double,N,N>> es(H);
    auto vals = es.eigenvalues().cwiseMax(0.0);
    return es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
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
    a << e1.dot(e1), d12, d12, e2.dot(e2);
    return a;
}

static Matrix<double, 4, 9> firstFFDeriv(const ShellMesh &mesh, int face)
{
    const Vector3i &tri = mesh.faces[face];
    Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
    Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];
    Matrix<double, 4, 9> D = Matrix<double, 4, 9>::Zero();
    D.block<1,3>(0, 0) = -2.0 * e1.transpose();
    D.block<1,3>(0, 3) =  2.0 * e1.transpose();
    D.block<1,3>(1, 0) = -(e1 + e2).transpose();
    D.block<1,3>(1, 3) =  e2.transpose();
    D.block<1,3>(1, 6) =  e1.transpose();
    D.row(2) = D.row(1);
    D.block<1,3>(3, 0) = -2.0 * e2.transpose();
    D.block<1,3>(3, 6) =  2.0 * e2.transpose();
    return D;
}

static void firstFFHessian(Matrix<double, 9, 9> ahess[4])
{
    for (int i = 0; i < 4; ++i) ahess[i].setZero();
    Matrix3d I3 = Matrix3d::Identity();
    ahess[0].block<3,3>(0,0) =  2*I3; ahess[0].block<3,3>(3,3) =  2*I3;
    ahess[0].block<3,3>(0,3) = -2*I3; ahess[0].block<3,3>(3,0) = -2*I3;
    ahess[1].block<3,3>(0,0) =  2*I3; ahess[1].block<3,3>(0,3) = -I3;
    ahess[1].block<3,3>(0,6) = -I3;   ahess[1].block<3,3>(3,0) = -I3;
    ahess[1].block<3,3>(3,6) =  I3;   ahess[1].block<3,3>(6,0) = -I3;
    ahess[1].block<3,3>(6,3) =  I3;
    ahess[2] = ahess[1];
    ahess[3].block<3,3>(0,0) =  2*I3; ahess[3].block<3,3>(6,6) =  2*I3;
    ahess[3].block<3,3>(0,6) = -2*I3; ahess[3].block<3,3>(6,0) = -2*I3;
}

// =============================================================================
// Second fundamental form (libshell MidedgeAverage) + derivative
// =============================================================================

Matrix2d secondFundamentalForm(const ShellMesh &mesh,
                               const std::vector<Vector3d> &fN, int face)
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d nc = fN[face];
    double II[3];
    for (int i = 0; i < 3; ++i) {
        int ip1 = (i+1)%3, ip2 = (i+2)%3;
        Vector3d qv = q[tri[ip1]] + q[tri[ip2]] - 2.0*q[tri[i]];
        int eid = mesh.faceEdges[face][i];
        const auto &r = mesh.edgeFaces[eid];
        int of = (r[0].face == face) ? r[1].face : r[0].face;
        if (of == -1) { II[i] = 0; continue; }
        Vector3d mv = fN[of] + nc;
        double mn = mv.norm();
        II[i] = (mn > 0) ? qv.dot(fN[of]) / mn : 0;
    }
    Matrix2d b;
    b << II[0]+II[1], II[0], II[0], II[0]+II[2];
    return b;
}

static Matrix<double, 4, 18> secondFFDeriv(
    const ShellMesh &mesh, const std::vector<Vector3d> &fN,
    int face, int oppVerts[3])
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d nc = fN[face];
    Vector3d e1 = q[tri[1]] - q[tri[0]], e2 = q[tri[2]] - q[tri[0]];

    Matrix<double,3,9> dnc = Matrix<double,3,9>::Zero();
    dnc.block<3,3>(0,0) = skew(q[tri[2]]-q[tri[1]]);
    dnc.block<3,3>(0,3) = -skew(e2);
    dnc.block<3,3>(0,6) = skew(e1);

    Matrix<double,1,18> dII[3];
    for (int i = 0; i < 3; ++i) dII[i].setZero();

    for (int i = 0; i < 3; ++i) {
        int ip1=(i+1)%3, ip2=(i+2)%3;
        Vector3d qv = q[tri[ip1]] + q[tri[ip2]] - 2.0*q[tri[i]];
        int eid = mesh.faceEdges[face][i];
        const auto &r = mesh.edgeFaces[eid];
        int of = (r[0].face==face) ? r[1].face : r[0].face;
        int olv = (r[0].face==face) ? r[1].localOppVtx : r[0].localOppVtx;
        if (of == -1) { oppVerts[i] = -1; continue; }
        oppVerts[i] = mesh.faces[of][olv];

        Vector3d no = fN[of], mv = no + nc;
        double mn = mv.norm();
        if (mn < 1e-16) continue;
        double IIv = qv.dot(no) / mn;

        // Part 1: qvec
        Vector3d nom = no / mn;
        dII[i].segment<3>(3*i)   += -2.0 * nom.transpose();
        dII[i].segment<3>(3*ip1) +=        nom.transpose();
        dII[i].segment<3>(3*ip2) +=        nom.transpose();

        // Opp face normal derivative
        int ov0=mesh.faces[of][olv], ov1=mesh.faces[of][(olv+1)%3], ov2=mesh.faces[of][(olv+2)%3];
        Vector3d oe1=q[ov1]-q[ov0], oe2=q[ov2]-q[ov0];
        Matrix3d dno0=skew(q[ov2]-q[ov1]), dno1=-skew(oe2), dno2=skew(oe1);

        auto findL = [&](int gv) { for(int j=0;j<3;++j) if(tri[j]==gv) return j; return -1; };
        int l1=findL(ov1), l2=findL(ov2);

        // Part 2: n_opp in numerator
        Vector3d qom = qv / mn;
        dII[i].segment<3>(9+3*i) += qom.transpose() * dno0;
        if (l1>=0) dII[i].segment<3>(3*l1) += qom.transpose() * dno1;
        if (l2>=0) dII[i].segment<3>(3*l2) += qom.transpose() * dno2;

        // Part 3: ||mvec|| denominator
        double c = -IIv / (mn*mn);
        RowVector3d mt = mv.transpose();
        dII[i].segment<3>(9+3*i) += c * mt * dno0;
        if (l1>=0) dII[i].segment<3>(3*l1) += c * mt * dno1;
        if (l2>=0) dII[i].segment<3>(3*l2) += c * mt * dno2;
        for (int j=0; j<3; ++j)
            dII[i].segment<3>(3*j) += c * mt * dnc.block<3,3>(0,3*j);
    }

    Matrix<double,4,18> bD;
    bD.row(0) = dII[0]+dII[1];
    bD.row(1) = dII[0];
    bD.row(2) = dII[0];
    bD.row(3) = dII[0]+dII[2];
    return bD;
}

// =============================================================================
// StVK gradient/Hessian helper (shared by elastic + damping terms)
// =============================================================================

// Given strain M = aBarInv*(X - Xbar) and derivative dX/dDOF (4×N),
// compute the StVK gradient (N×1) and Gauss-Newton Hessian (N×N).
// coef is the energy scaling factor. Projects Hessian to PSD.
template<int N>
static void stvkGradHess(
    const Matrix2d &M, const Matrix2d &aBarInv,
    const Matrix<double,4,N> &dXdDOF,
    double alpha, double beta, double coef,
    Matrix<double,N,1> &grad, Matrix<double,N,N> &hess)
{
    // Stress (4-vector)
    Matrix2d stress = alpha * M.trace() * aBarInv + 2.0 * beta * M * aBarInv;
    Map<Vector4d> sv(stress.data());
    grad = coef * dXdDOF.transpose() * sv;

    // Gauss-Newton Hessian: coef * (∇r1 ∇r1^T + ∇r2 ∇r2^T)
    Map<const Vector4d> ainv_v(aBarInv.data());
    Matrix<double,N,1> dr1 = std::sqrt(0.5*alpha) * dXdDOF.transpose() * ainv_v;

    double trM2 = (M*M).trace();
    Matrix2d Mainv = M * aBarInv;
    Map<Vector4d> mainv_v(Mainv.data());
    Matrix<double,N,1> dr2;
    if (trM2 > 1e-16)
        dr2 = (beta / std::sqrt(beta * trM2)) * dXdDOF.transpose() * mainv_v;
    else
        dr2.setZero();

    hess = coef * (dr1 * dr1.transpose() + dr2 * dr2.transpose());
}

// =============================================================================
// Per-face stretching
// =============================================================================

StretchingData stretchingPerFace(const ShellMesh &mesh,
                                 const ShellRestState &rest,
                                 const MaterialParams &mat, int face)
{
    StretchingData r;
    double alpha = mat.alpha(), beta = mat.beta();
    Matrix2d a = firstFundamentalForm(mesh, face);
    Matrix2d aBI = rest.aBar[face].inverse();
    Matrix2d M = aBI * a - Matrix2d::Identity();
    double coef = 0.25 * mat.thickness * rest.restArea[face];

    r.energy = coef * svNormSq(M, alpha, beta);

    Matrix<double,4,9> aD = firstFFDeriv(mesh, face);
    Map<Vector4d> sv((Matrix2d(alpha*M.trace()*aBI + 2*beta*M*aBI)).data());
    r.gradient = coef * aD.transpose() * sv;

    // Full Hessian (3 terms)
    Matrix<double,9,9> ah[4]; firstFFHessian(ah);
    Map<const Vector4d> abiv(aBI.data());
    Matrix<double,9,1> in1 = aD.transpose() * abiv;
    r.hessian = coef * alpha * in1 * in1.transpose();

    Matrix2d Mai = M * aBI;
    for (int k = 0; k < 4; ++k) {
        double s = alpha*M.trace()*aBI.data()[k] + 2*beta*Mai.data()[k];
        r.hessian += coef * s * ah[k];
    }

    Matrix<double,1,9> i00=aBI(0,0)*aD.row(0)+aBI(0,1)*aD.row(1);
    Matrix<double,1,9> i01=aBI(0,0)*aD.row(2)+aBI(0,1)*aD.row(3);
    Matrix<double,1,9> i10=aBI(1,0)*aD.row(0)+aBI(1,1)*aD.row(1);
    Matrix<double,1,9> i11=aBI(1,0)*aD.row(2)+aBI(1,1)*aD.row(3);
    r.hessian += coef*2*beta*(i00.transpose()*i00 + i01.transpose()*i10
                             + i10.transpose()*i01 + i11.transpose()*i11);

    // Project stretching Hessian to PSD. The 9×9 eigendecomposition is
    // cheap (~500 flops) unlike the 18×18 bending case.
    r.hessian = projectPSD<9>(r.hessian);
    return r;
}

// =============================================================================
// Per-face bending
// =============================================================================

BendingData bendingPerFace(const ShellMesh &mesh, const ShellRestState &rest,
                           const MaterialParams &mat,
                           const std::vector<Vector3d> &fN, int face)
{
    BendingData r;
    double alpha = mat.alpha(), beta = mat.beta();
    double h3_12 = mat.thickness*mat.thickness*mat.thickness / 12.0;
    double coef = h3_12 * rest.restArea[face];

    int ov[3];
    Matrix<double,4,18> bD = secondFFDeriv(mesh, fN, face, ov);
    for (int i=0;i<3;++i) r.vertIdx[i] = mesh.faces[face][i];
    for (int i=0;i<3;++i) r.vertIdx[3+i] = ov[i];

    Matrix2d b = secondFundamentalForm(mesh, fN, face);
    Matrix2d aBI = rest.aBar[face].inverse();
    Matrix2d Mb = aBI * (b - rest.bBar[face]);

    r.energy = coef * svNormSq(Mb, alpha, beta);

    // Gradient + inexact Hessian via shared helper
    stvkGradHess<18>(Mb, aBI, bD, alpha, beta, coef, r.gradient, r.hessian);

    return r;
}

// =============================================================================
// Total energy (scalar only, for line search / diagnostics)
// =============================================================================

// Energy-only evaluation. Accepts optional precomputed normals to avoid
// redundant computation during line search.
double totalEnergy(const ShellMesh &mesh, const ShellRestState &rest,
                   const MaterialParams &mat,
                   const std::vector<Vector3d> *cachedNormals)
{
    double alpha = mat.alpha(), beta = mat.beta();
    double h3_12 = mat.thickness*mat.thickness*mat.thickness / 12.0;
    int nF = mesh.numFaces();
    std::vector<Vector3d> fNLocal;
    if (!cachedNormals) { computeFaceNormals(mesh, fNLocal); cachedNormals = &fNLocal; }
    const auto &fN = *cachedNormals;
    double tot = 0;
    #pragma omp parallel for reduction(+:tot) schedule(static)
    for (int f = 0; f < nF; ++f) {
        Matrix2d a = firstFundamentalForm(mesh, f);
        Matrix2d aBI = rest.aBar[f].inverse();
        tot += 0.25*mat.thickness*rest.restArea[f]*svNormSq(aBI*a-Matrix2d::Identity(), alpha, beta);
        Matrix2d b = secondFundamentalForm(mesh, fN, f);
        tot += h3_12*rest.restArea[f]*svNormSq(aBI*(b-rest.bBar[f]), alpha, beta);
    }
    return tot;
}

// =============================================================================
// Global assembly (elastic + Kelvin-Voigt damping)
// =============================================================================

void assembleGradientAndHessian(
    ShellMesh &mesh, const ShellRestState &rest, const MaterialParams &mat,
    const DampingState &damp, double dt,
    VectorXd &grad, std::vector<Triplet<double>> &trip)
{
    int n = mesh.numVerts(), nF = mesh.numFaces(), dim = 3*n;
    grad.setZero(dim);
    trip.clear();

    std::vector<Vector3d> fN; computeFaceNormals(mesh, fN);

    const int nThreads = omp_get_max_threads();
    std::vector<VectorXd> tGrad(nThreads, VectorXd::Zero(dim));
    std::vector<std::vector<Triplet<double>>> tTrip(nThreads);

    auto scatterBlock = [](std::vector<Triplet<double>> &lt,
                           const int *vIdx, int nv,
                           const auto &hess) {
        for (int i = 0; i < nv; ++i) {
            if (vIdx[i] < 0) continue;
            for (int j = 0; j < nv; ++j) {
                if (vIdx[j] < 0) continue;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        lt.emplace_back(3*vIdx[i]+a, 3*vIdx[j]+b,
                                        hess(3*i+a, 3*j+b));
            }
        }
    };

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto &lg = tGrad[tid];
        auto &lt = tTrip[tid];

    #pragma omp for schedule(dynamic)
    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = mesh.faces[f];
        int triIdx[3] = {tri[0], tri[1], tri[2]};

        auto sd = stretchingPerFace(mesh, rest, mat, f);
        for (int i=0;i<3;++i) lg.segment<3>(3*tri[i]) += sd.gradient.segment<3>(3*i);
        scatterBlock(lt, triIdx, 3, sd.hessian);

        auto bd = bendingPerFace(mesh, rest, mat, fN, f);
        for (int i=0;i<6;++i) if (bd.vertIdx[i]>=0)
            lg.segment<3>(3*bd.vertIdx[i]) += bd.gradient.segment<3>(3*i);
        scatterBlock(lt, bd.vertIdx, 6, bd.hessian);
    }
    }

    for (int t = 0; t < nThreads; ++t) {
        grad += tGrad[t];
        trip.insert(trip.end(), tTrip[t].begin(), tTrip[t].end());
    }
}

// =============================================================================
// Implicit Euler with Newton (paper Section 5)
// =============================================================================

void stepImplicitEuler(
    ShellMesh &mesh, const ShellRestState &rest, const MaterialParams &mat,
    const Vector3d &gravity, std::vector<double> &masses,
    std::vector<Vector3d> &velocities, DampingState &damp,
    double dt, int maxIters, double tol)
{
    const int n = mesh.numVerts(), dim = 3*n;
    const double inv_dt2 = 1.0 / (dt*dt);

    auto pos0 = mesh.vertices;
    std::vector<Vector3d> xTilde(n);
    for (int i=0; i<n; ++i)
        xTilde[i] = pos0[i] + dt*velocities[i] + dt*dt*gravity;
    mesh.vertices = xTilde;

    const int numFaces = mesh.numFaces();
    std::vector<Vector3d> restNormals(numFaces);
    for (int f = 0; f < numFaces; ++f) {
        const auto &t = mesh.faces[f];
        restNormals[f] = (pos0[t[1]]-pos0[t[0]]).cross(pos0[t[2]]-pos0[t[0]]);
    }

    for (int iter = 0; iter < maxIters; ++iter) {
        VectorXd eGrad;
        std::vector<Triplet<double>> hTrip;
        assembleGradientAndHessian(mesh, rest, mat, damp, dt, eGrad, hTrip);

        VectorXd g(dim);
        for (int i=0; i<n; ++i) {
            Vector3d dx = mesh.vertices[i] - xTilde[i];
            g.segment<3>(3*i) = masses[i]*inv_dt2*dx + Vector3d(eGrad.segment<3>(3*i));
        }
        if (g.norm() < tol) break;

        auto posSave = mesh.vertices;

        auto incrPotential = [&]() -> double {
            std::vector<Vector3d> curFN; computeFaceNormals(mesh, curFN);
            return totalEnergy(mesh, rest, mat, &curFN);
        };

        // Factorize with progressive regularization if needed.
        VectorXd dx;
        bool solved = false;
        for (double reg = 0.0; reg <= 16.0; reg = (reg == 0.0) ? 1.0 : reg * 2.0) {
            std::vector<Triplet<double>> sTrip;
            sTrip.reserve(hTrip.size() + dim);
            for (int i = 0; i < n; ++i) {
                double diag = (1.0 + reg) * masses[i] * inv_dt2;
                for (int a = 0; a < 3; ++a)
                    sTrip.emplace_back(3*i+a, 3*i+a, diag);
            }
            for (auto &t : hTrip)
                sTrip.emplace_back(t.row(), t.col(), t.value());

            SparseMatrix<double> H(dim, dim);
            H.setFromTriplets(sTrip.begin(), sTrip.end());

            SimplicialLDLT<SparseMatrix<double>> solver(H);
            if (solver.info() != Eigen::Success) continue;
            dx = solver.solve(-g);
            if (solver.info() == Eigen::Success) { solved = true; break; }
        }
        if (!solved) break;

        // Backtrack only for triangle inversion.
        double alpha_ls = 1.0;
        for (int ls = 0; ls < 20; ++ls) {
            for (int i = 0; i < n; ++i)
                mesh.vertices[i] = posSave[i] + alpha_ls * Vector3d(dx.segment<3>(3*i));

            bool inverted = false;
            for (int f = 0; f < numFaces; ++f) {
                const auto &t = mesh.faces[f];
                Vector3d nn = (mesh.vertices[t[1]] - mesh.vertices[t[0]])
                              .cross(mesh.vertices[t[2]] - mesh.vertices[t[0]]);
                if (nn.dot(restNormals[f]) <= 0) { inverted = true; break; }
            }
            if (!inverted) break;
            alpha_ls *= 0.5;
        }
    }

    // Zero velocity: quasi-static stepping. Full momentum (v = dx/dt)
    // causes overshoot with the paper's tiny η. Re-enable when Kelvin-
    // Voigt damping is tuned to dissipate kinetic energy.
    for (int i=0; i<n; ++i)
        velocities[i] = Vector3d::Zero();

    // Store current fundamental forms for next step's damping
    // (only when damping is actually active).
    if (mat.viscosity / mat.young > 1e-10) {
        int nF = mesh.numFaces();
        damp.aPrev.resize(nF);
        damp.bPrev.resize(nF);
        std::vector<Vector3d> fN; computeFaceNormals(mesh, fN);
        for (int f=0; f<nF; ++f) {
            damp.aPrev[f] = firstFundamentalForm(mesh, f);
            damp.bPrev[f] = secondFundamentalForm(mesh, fN, f);
        }
    }
}

// =============================================================================
// Lumped mass
// =============================================================================

void computeLumpedMasses(const ShellMesh &mesh, const ShellRestState &rest,
                         const MaterialParams &mat, std::vector<double> &masses)
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

void verifyForceGradient(ShellMesh &mesh, const ShellRestState &rest,
                         const MaterialParams &mat, const DampingState &damp,
                         double dt, double eps)
{
    if (mesh.vertices.empty()) return;
    int n = mesh.numVerts();
    auto saved = mesh.vertices;

    for (int i=0; i<n; ++i)
        mesh.vertices[i] += Vector3d(0.01*std::sin(0.7*i+1),
                                      0.01*std::cos(1.3*i+0.4),
                                      0.01*std::sin(0.5*i-0.2));

    VectorXd grad;
    std::vector<Triplet<double>> hTrip;
    assembleGradientAndHessian(mesh, rest, mat, damp, dt, grad, hTrip);

    double wAbs=0, wRel=0; int wI=-1, wK=-1;
    for (int i=0; i<n; ++i) for (int k=0; k<3; ++k) {
        double orig = mesh.vertices[i][k];
        mesh.vertices[i][k] = orig+eps; double Ep = totalEnergy(mesh,rest,mat);
        mesh.vertices[i][k] = orig-eps; double Em = totalEnergy(mesh,rest,mat);
        mesh.vertices[i][k] = orig;
        double nf = (Ep-Em)/(2*eps), af = grad[3*i+k];
        double ae = std::abs(nf-af), den = std::max({std::abs(nf),std::abs(af),1e-12});
        if (ae > wAbs) { wAbs=ae; wRel=ae/den; wI=i; wK=k; }
    }
    std::cout << "[FD check] |abs|=" << wAbs << " rel=" << wRel
              << " v" << wI << " axis" << wK << std::endl;
    mesh.vertices = saved;
}
