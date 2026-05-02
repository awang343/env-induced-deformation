#pragma once

#include "shell_mesh.h"
#include "energy.h"
#include <Eigen/Dense>
#include <vector>

void stepImplicitEuler(
    ShellMesh &mesh,
    const ShellRestState &rest,
    const MaterialParams &mat,
    std::vector<double> &masses,
    std::vector<Eigen::Vector3d> &velocities,
    double dt,
    int maxIters = 10,
    double tol = 1e-6);
