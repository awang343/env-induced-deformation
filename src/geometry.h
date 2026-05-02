#pragma once

#include "shell_mesh.h"
#include <Eigen/Dense>
#include <vector>

// Fundamental forms and their derivatives.

Eigen::Matrix3d skew(const Eigen::Vector3d &v);

double svNormSq(const Eigen::Matrix2d &M, double alpha, double beta);

template<int N>
Eigen::Matrix<double,N,N> projectPSD(const Eigen::Matrix<double,N,N> &H);

void computeFaceNormals(const ShellMesh &mesh,
                        std::vector<Eigen::Vector3d> &normals);

Eigen::Matrix2d firstFundamentalForm(const ShellMesh &mesh, int face);
Eigen::Matrix<double, 4, 9> firstFFDeriv(const ShellMesh &mesh, int face);
void firstFFHessian(Eigen::Matrix<double, 9, 9> ahess[4]);

Eigen::Matrix2d secondFundamentalForm(
    const ShellMesh &mesh,
    const std::vector<Eigen::Vector3d> &faceNormals,
    int face);
Eigen::Matrix<double, 4, 18> secondFFDeriv(
    const ShellMesh &mesh,
    const std::vector<Eigen::Vector3d> &faceNormals,
    int face, int oppVerts[3]);
