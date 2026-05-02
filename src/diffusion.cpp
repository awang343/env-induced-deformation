#include "diffusion.h"

#include <Eigen/Sparse>
#ifdef HAVE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

using namespace Eigen;

#ifdef HAVE_CHOLMOD
using SparseSolver = CholmodSupernodalLLT<SparseMatrix<double>>;
#else
using SparseSolver = SimplicialLDLT<SparseMatrix<double>>;
#endif

void diffuseMoisture(const ShellMesh &mesh,
                     const ShellRestState &rest,
                     const MaterialParams &mat,
                     double dt, double diffusivity,
                     std::vector<double> &mPlus,
                     std::vector<double> &mMinus,
                     const std::vector<double> &sPlus,
                     const std::vector<double> &sMinus)
{
    const int nV = mesh.numVerts();
    const int nF = mesh.numFaces();
    const double h = mat.thickness;
    const int dim = 2 * nV;

    std::vector<Triplet<double>> trips;

    for (int f = 0; f < nF; ++f) {
        const auto &tri = mesh.faces[f];
        double area = rest.restArea[f];
        const Matrix2d &ab = rest.aBar[f];

        double mDiag = area / 6.0;
        double mOff  = area / 12.0;

        double cot0 = ab(0,1) / (2.0 * area);
        double cot1 = (ab(0,0) - ab(0,1)) / (2.0 * area);
        double cot2 = (ab(1,1) - ab(0,1)) / (2.0 * area);

        double kEdge[3] = {-cot0 / 2.0, -cot1 / 2.0, -cot2 / 2.0};
        int edgeVerts[3][2] = {{tri[1],tri[2]}, {tri[0],tri[2]}, {tri[0],tri[1]}};

        int localVerts[3] = {tri[0], tri[1], tri[2]};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                int va = localVerts[a], vb = localVerts[b];
                double m2d = (a == b) ? mDiag : mOff;

                trips.emplace_back(va,      vb,      h/3.0 * m2d);
                trips.emplace_back(va,      nV+vb,   h/6.0 * m2d);
                trips.emplace_back(nV+va,   vb,      h/6.0 * m2d);
                trips.emplace_back(nV+va,   nV+vb,   h/3.0 * m2d);

                double kz = m2d / h;
                trips.emplace_back(va,      vb,      dt * diffusivity * kz);
                trips.emplace_back(va,      nV+vb,   dt * diffusivity * (-kz));
                trips.emplace_back(nV+va,   vb,      dt * diffusivity * (-kz));
                trips.emplace_back(nV+va,   nV+vb,   dt * diffusivity * kz);
            }
        }

        for (int e = 0; e < 3; ++e) {
            int va = edgeVerts[e][0], vb = edgeVerts[e][1];
            double k = kEdge[e];

            trips.emplace_back(va,    vb,    dt * diffusivity * h/3.0 * k);
            trips.emplace_back(vb,    va,    dt * diffusivity * h/3.0 * k);
            trips.emplace_back(va,    nV+vb, dt * diffusivity * h/6.0 * k);
            trips.emplace_back(vb,    nV+va, dt * diffusivity * h/6.0 * k);
            trips.emplace_back(nV+va, vb,    dt * diffusivity * h/6.0 * k);
            trips.emplace_back(nV+vb, va,    dt * diffusivity * h/6.0 * k);
            trips.emplace_back(nV+va, nV+vb, dt * diffusivity * h/3.0 * k);
            trips.emplace_back(nV+vb, nV+va, dt * diffusivity * h/3.0 * k);

            trips.emplace_back(va,    va,    dt * diffusivity * h/3.0 * (-k));
            trips.emplace_back(vb,    vb,    dt * diffusivity * h/3.0 * (-k));
            trips.emplace_back(va,    nV+va, dt * diffusivity * h/6.0 * (-k));
            trips.emplace_back(vb,    nV+vb, dt * diffusivity * h/6.0 * (-k));
            trips.emplace_back(nV+va, va,    dt * diffusivity * h/6.0 * (-k));
            trips.emplace_back(nV+vb, vb,    dt * diffusivity * h/6.0 * (-k));
            trips.emplace_back(nV+va, nV+va, dt * diffusivity * h/3.0 * (-k));
            trips.emplace_back(nV+vb, nV+vb, dt * diffusivity * h/3.0 * (-k));
        }
    }

    SparseMatrix<double> A(dim, dim);
    A.setFromTriplets(trips.begin(), trips.end());

    VectorXd m_old(dim), rhs(dim);
    for (int i = 0; i < nV; ++i) {
        m_old[i]    = mPlus[i]  + (sPlus.empty()  ? 0.0 : dt * sPlus[i]);
        m_old[nV+i] = mMinus[i] + (sMinus.empty() ? 0.0 : dt * sMinus[i]);
    }

    std::vector<Triplet<double>> mTrips;
    for (int f = 0; f < nF; ++f) {
        const auto &tri = mesh.faces[f];
        double area = rest.restArea[f];
        double mDiag = area / 6.0, mOff = area / 12.0;
        int lv[3] = {tri[0], tri[1], tri[2]};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double m2d = (a == b) ? mDiag : mOff;
                mTrips.emplace_back(lv[a],    lv[b],    h/3.0 * m2d);
                mTrips.emplace_back(lv[a],    nV+lv[b], h/6.0 * m2d);
                mTrips.emplace_back(nV+lv[a], lv[b],    h/6.0 * m2d);
                mTrips.emplace_back(nV+lv[a], nV+lv[b], h/3.0 * m2d);
            }
    }
    SparseMatrix<double> MG(dim, dim);
    MG.setFromTriplets(mTrips.begin(), mTrips.end());
    rhs = MG * m_old;

    SparseSolver solver;
    solver.compute(A);
    VectorXd m_new = solver.solve(rhs);

    for (int i = 0; i < nV; ++i) {
        mPlus[i]  = m_new[i];
        mMinus[i] = m_new[nV+i];
    }
}
