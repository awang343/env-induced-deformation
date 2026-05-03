#include "implicit_euler.h"
#include "geometry.h"

#include <Eigen/Sparse>
#ifdef HAVE_CHOLMOD
#include <Eigen/CholmodSupport>
#endif

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Eigen;

#ifdef HAVE_CHOLMOD
using SparseSolver = CholmodSupernodalLLT<SparseMatrix<double>>;
#else
using SparseSolver = SimplicialLDLT<SparseMatrix<double>>;
#endif

void stepImplicitEuler(
    ShellMesh &mesh, const ShellRestState &rest, const MaterialParams &mat,
    std::vector<double> &masses,
    std::vector<Vector3d> &velocities,
    double dt, int maxIters, double tol)
{
    const int n = mesh.numVerts(), dim = 3*n;
    const double inv_dt2 = 1.0 / (dt*dt);

    static SparseMatrix<double> H;
    static SparseSolver solver;
    static bool patternBuilt = false;

    if (!patternBuilt) {
        VectorXd dummyGrad;
        std::vector<Triplet<double>> dummyTrip;
        assembleGradientAndHessian(mesh, rest, mat, dummyGrad, dummyTrip);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
                dummyTrip.emplace_back(3*i+a, 3*i+a, 1.0);
        H.resize(dim, dim);
        H.setFromTriplets(dummyTrip.begin(), dummyTrip.end());
        H.makeCompressed();
        solver.analyzePattern(H);
        patternBuilt = true;
    }

    auto pos0 = mesh.vertices;
    std::vector<Vector3d> xTilde(n);
    for (int i=0; i<n; ++i)
        xTilde[i] = pos0[i] + dt*velocities[i];
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
        assembleGradientAndHessian(mesh, rest, mat, eGrad, hTrip);

        VectorXd g(dim);
        for (int i=0; i<n; ++i) {
            Vector3d dx = mesh.vertices[i] - xTilde[i];
            g.segment<3>(3*i) = masses[i]*inv_dt2*dx + Vector3d(eGrad.segment<3>(3*i));
        }
        if (g.norm() < tol) break;

        auto posSave = mesh.vertices;

        // Scatter elastic Hessian once, save for reg loop.
        std::fill(H.valuePtr(), H.valuePtr() + H.nonZeros(), 0.0);
        for (auto &t : hTrip)
            H.coeffRef(t.row(), t.col()) += t.value();
        std::vector<double> elasticVals(H.valuePtr(), H.valuePtr() + H.nonZeros());

        VectorXd dx;
        bool solved = false;
        for (double reg = 0.0; reg <= 16.0; reg = (reg == 0.0) ? 1.0 : reg * 2.0) {
            std::copy(elasticVals.begin(), elasticVals.end(), H.valuePtr());
            for (int i = 0; i < n; ++i) {
                double diag = (1.0 + reg) * masses[i] * inv_dt2;
                for (int a = 0; a < 3; ++a)
                    H.coeffRef(3*i+a, 3*i+a) += diag;
            }

            solver.factorize(H);
            if (solver.info() != Eigen::Success) continue;
            dx = solver.solve(-g);
            if (solver.info() == Eigen::Success) { solved = true; break; }
        }
        if (!solved) break;

        // Backtrack for inversion + energy decrease.
        auto elasticEnergy = [&]() -> double {
            std::vector<Vector3d> curFN; computeFaceNormals(mesh, curFN);
            return totalEnergy(mesh, rest, mat, &curFN);
        };
        double E0 = elasticEnergy();

        double alpha_ls = 1.0;
        bool accepted = false;
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
            if (!inverted && elasticEnergy() < E0) { accepted = true; break; }
            alpha_ls *= 0.5;
        }
        if (!accepted) mesh.vertices = posSave;
    }

    for (int i=0; i<n; ++i)
        velocities[i] = 0.5 * (mesh.vertices[i] - pos0[i]) / dt;  // TODO: replace with Kelvin-Voigt damping
}
