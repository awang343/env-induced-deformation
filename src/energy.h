#pragma once

#include "shell_mesh.h"
#include "geometry.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

struct MaterialParams
{
    double thickness  = 1e-4;
    double young      = 2e9;
    double poisson    = 0.3;
    double density    = 250.0;

    double alpha() const { return young * poisson / (1.0 - poisson * poisson); }
    double beta()  const { return young / (2.0 * (1.0 + poisson)); }
};

struct ShellRestState
{
    std::vector<Eigen::Matrix2d> aBar;
    std::vector<Eigen::Matrix2d> bBar;
    std::vector<double>          restArea;
};

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
                           const std::vector<Eigen::Vector3d> &faceNormals,
                           int face);

double totalEnergy(const ShellMesh &mesh,
                   const ShellRestState &rest,
                   const MaterialParams &mat,
                   const std::vector<Eigen::Vector3d> *cachedNormals = nullptr);

void assembleGradientAndHessian(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    Eigen::VectorXd &gradient,
    std::vector<Eigen::Triplet<double>> &hessianTriplets);

void computeLumpedMasses(const ShellMesh &mesh,
                         const ShellRestState &rest,
                         const MaterialParams &mat,
                         std::vector<double> &masses);
