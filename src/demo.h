#pragma once

#include "shell_mesh.h"
#include "shell_energy.h"
#include <Eigen/Dense>
#include <vector>

// ---- Stereographic disk-to-sphere (Figure 3) ----

// Disk-to-sphere via stretching: āBar from sphere, b̄ = 0.
void initSphereStretching(ShellMesh &mesh,
                          const std::vector<Eigen::Matrix2d> &a0,
                          ShellRestState &rest,
                          int seed = 42, double perturbScale = 0.05);

// Disk-to-sphere via bending: āBar = flat, b̄ from sphere curvature.
void initSphereBending(ShellMesh &mesh,
                       const std::vector<Eigen::Matrix2d> &a0,
                       ShellRestState &rest,
                       int seed = 42, double perturbScale = 0.05);

// Isotropic growth: āBar = s² · a₀, b̄ = 0.
// Uniform scaling of the rest metric — the mesh wants to expand
// uniformly by factor s but is constrained by its geometry.
void initIsotropicGrowth(ShellMesh &mesh,
                         const std::vector<Eigen::Matrix2d> &a0,
                         ShellRestState &rest,
                         double growthFactor = 2.0,
                         int seed = 42, double perturbScale = 0.01);

// Cylinder curling via pure bending: āBar = flat, b̄ = κ in x-direction.
void initCylinderDemo(ShellMesh &mesh,
                      const std::vector<Eigen::Matrix2d> &a0,
                      ShellRestState &rest,
                      double kappa = 1.0,
                      int seed = 42, double perturbScale = 0.01);
