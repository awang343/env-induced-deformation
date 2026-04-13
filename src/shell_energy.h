#pragma once

#include "shell_mesh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// Material parameters for the Koiter shell model.
struct MaterialParams
{
    double thickness  = 1e-4;   // h
    double young      = 2e9;    // E
    double poisson    = 0.3;    // nu
    double density    = 250.0;  // rho

    double alpha() const { return young * poisson / (1.0 - poisson * poisson); }
    double beta()  const { return young / (2.0 * (1.0 + poisson)); }
};

// Rest state carried per-face.
struct ShellRestState
{
    std::vector<Eigen::Matrix2d> aBar;
    std::vector<Eigen::Matrix2d> bBar;
    std::vector<double>          restArea;   // sqrt(det aBar) / 2
};

// ---- Fundamental forms ----

Eigen::Matrix2d firstFundamentalForm(const ShellMesh &mesh, int face);

// Uses libshell's MidedgeAverageFormulation:
//   II[i] = qvec[i] · n_opp[i] / ||mvec[i]||
//   b = [[II0+II1, II0], [II0, II0+II2]]
// Requires precomputed unnormalized face normals.
Eigen::Matrix2d secondFundamentalForm(
    const ShellMesh &mesh,
    const std::vector<Eigen::Vector3d> &faceNormalsUnnorm,
    int face);

// Unnormalized face normals (cross product, not normalized).
void computeFaceNormals(const ShellMesh &mesh,
                        std::vector<Eigen::Vector3d> &normals);

// ---- Per-face derivative results ----

struct StretchingData
{
    double energy;
    Eigen::Matrix<double, 9, 1> gradient;
    Eigen::Matrix<double, 9, 9> hessian;
};

struct BendingData
{
    double energy;
    Eigen::Matrix<double, 18, 1> gradient;
    Eigen::Matrix<double, 18, 18> hessian;   // inexact (Gauss-Newton)
    int vertIdx[6];  // global vertex indices: [tri0,tri1,tri2,opp0,opp1,opp2], -1 if boundary
};

// Compute per-face stretching energy, gradient (9-DOF), and Hessian (9×9).
StretchingData stretchingPerFace(const ShellMesh &mesh,
                                 const ShellRestState &rest,
                                 const MaterialParams &mat,
                                 int face);

// Compute per-face bending energy, gradient (18-DOF), and inexact Hessian (18×18).
// Uses the MidedgeAverage formulation with analytical ∂b/∂DOF.
BendingData bendingPerFace(const ShellMesh &mesh,
                           const ShellRestState &rest,
                           const MaterialParams &mat,
                           const std::vector<Eigen::Vector3d> &faceNormalsUnnorm,
                           int face);

// ---- Global assembly ----

double totalEnergy(const ShellMesh &mesh,
                   const ShellRestState &rest,
                   const MaterialParams &mat);

void assembleGradientAndHessian(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    const Eigen::Vector3d &gravity,
    const std::vector<double> &masses,
    Eigen::VectorXd &gradient,
    std::vector<Eigen::Triplet<double>> &hessianTriplets);

// ---- Implicit Euler with Newton ----

void stepImplicitEuler(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    const Eigen::Vector3d &gravity,
    std::vector<double> &masses,
    std::vector<Eigen::Vector3d> &velocities,
    double dt,
    int maxIters = 5,
    double tol = 1e-6);

// ---- Lumped mass ----

void computeLumpedMasses(const ShellMesh &mesh,
                         const ShellRestState &rest,
                         const MaterialParams &mat,
                         std::vector<double> &masses);

// ---- Diagnostics ----

void verifyForceGradient(ShellMesh &mesh,
                         const ShellRestState &rest,
                         const MaterialParams &mat,
                         const Eigen::Vector3d &gravity,
                         const std::vector<double> &masses,
                         double eps = 1e-6);
