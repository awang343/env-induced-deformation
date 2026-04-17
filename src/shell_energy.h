#pragma once

#include "shell_mesh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// Material parameters for the Koiter shell model (paper Table 1).
struct MaterialParams
{
    double thickness  = 1e-4;    // h    (m)
    double young      = 2e9;     // E    (Pa)
    double poisson    = 0.3;     // nu
    double density    = 250.0;   // rho  (kg/m³)
    double viscosity  = 5e-13;   // eta  Kelvin-Voigt damping (Pa·s)

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

// Per-face history for Kelvin-Voigt damping (previous timestep forms).
struct DampingState
{
    std::vector<Eigen::Matrix2d> aPrev;
    std::vector<Eigen::Matrix2d> bPrev;
};

// ---- Fundamental forms ----

Eigen::Matrix2d firstFundamentalForm(const ShellMesh &mesh, int face);

Eigen::Matrix2d secondFundamentalForm(
    const ShellMesh &mesh,
    const std::vector<Eigen::Vector3d> &faceNormalsUnnorm,
    int face);

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
    Eigen::Matrix<double, 18, 18> hessian;
    int vertIdx[6];
};

StretchingData stretchingPerFace(const ShellMesh &mesh,
                                 const ShellRestState &rest,
                                 const MaterialParams &mat,
                                 int face);

BendingData bendingPerFace(const ShellMesh &mesh,
                           const ShellRestState &rest,
                           const MaterialParams &mat,
                           const std::vector<Eigen::Vector3d> &faceNormalsUnnorm,
                           int face);

// ---- Global ----

double totalEnergy(const ShellMesh &mesh,
                   const ShellRestState &rest,
                   const MaterialParams &mat,
                   const std::vector<Eigen::Vector3d> *cachedNormals = nullptr);

void assembleGradientAndHessian(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    const DampingState &damp,
    double dt,
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
    DampingState &damp,
    double dt,
    int maxIters = 10,
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
                         const DampingState &damp,
                         double dt,
                         double eps = 1e-6);
