#include "energy.h"
#include "geometry.h"

#include <omp.h>
#include <cmath>

using namespace Eigen;

// =============================================================================
// StVK gradient/Hessian helper (shared by elastic + damping terms)
// =============================================================================

template<int N>
static void stvkGradHess(
    const Matrix2d &M, const Matrix2d &aBarInv,
    const Matrix<double,4,N> &dXdDOF,
    double alpha, double beta, double coef,
    Matrix<double,N,1> &grad, Matrix<double,N,N> &hess)
{
    Matrix2d stress = alpha * M.trace() * aBarInv + 2.0 * beta * M * aBarInv;
    Map<Vector4d> sv(stress.data());
    grad = coef * dXdDOF.transpose() * sv;

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

    stvkGradHess<18>(Mb, aBI, bD, alpha, beta, coef, r.gradient, r.hessian);

    return r;
}

// =============================================================================
// Total energy (scalar only, for line search / diagnostics)
// =============================================================================

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
// Global assembly
// =============================================================================

void assembleGradientAndHessian(
    ShellMesh &mesh, const ShellRestState &rest, const MaterialParams &mat,
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
