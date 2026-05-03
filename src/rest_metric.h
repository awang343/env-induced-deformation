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

// Swelling: āBar = a₀, b̄ = b₀ (initial geometry preserved).
// Rest forms will be updated dynamically from moisture scalars.
void initSwelling(ShellMesh &mesh,
                  const std::vector<Eigen::Matrix2d> &a0,
                  ShellRestState &rest,
                  int seed = 42, double perturbScale = 0.01);

// Cylinder curling via pure bending: āBar = flat, b̄ = κ in x-direction.
void initCylinderDemo(ShellMesh &mesh,
                      const std::vector<Eigen::Matrix2d> &a0,
                      ShellRestState &rest,
                      double kappa = 1.0,
                      int seed = 42, double perturbScale = 0.01);

// Sphere target: map any mesh to a unit sphere by normalizing vertices.
// Sets ā and b̄ from the sphere geometry. Works on any genus-0 mesh.
void initSphereTarget(ShellMesh &mesh,
                      const std::vector<Eigen::Matrix2d> &a0,
                      ShellRestState &rest,
                      double radius = 1.0,
                      int seed = 42, double perturbScale = 0.01);

// ---------- Swelling rest form updates (paper Section 2, page 6) ----------

// Update ā, b̄ from per-vertex moisture using linear differential swelling.
// g⁺ = (1+m⁺μ)²(a⁰ − hb⁰),  g⁻ = (1+m⁻μ)²(a⁰ + hb⁰)
// ā = (g⁺+g⁻)/2,  b̄ = (g⁻−g⁺)/(2h)
void updateRestFormsLinear(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Eigen::Matrix2d> &a0,
    const std::vector<Eigen::Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    double h, double mu);

// Update ā, b̄ from per-vertex moisture using piecewise constant swelling.
// g⁺ = (1+m⁺μ)²(a⁰ − ⅔hb⁰),  g⁻ = (1+m⁻μ)²(a⁰ + ⅔hb⁰)
// ā = (g⁺+g⁻)/2,  b̄ = 3/(4h)·(g⁻−g⁺)
void updateRestFormsPiecewise(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Eigen::Matrix2d> &a0,
    const std::vector<Eigen::Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    double h, double mu);

// Update ā, b̄ from per-vertex moisture using machine direction swelling.
// Anisotropic: μ along fibers, μ⊥ across fibers.
// Requires per-face machine direction d in barycentric coordinates.
void updateRestFormsMachine(
    const ShellMesh &mesh, ShellRestState &rest,
    const std::vector<Eigen::Matrix2d> &a0,
    const std::vector<Eigen::Matrix2d> &b0,
    const std::vector<double> &mPlus,
    const std::vector<double> &mMinus,
    const std::vector<Eigen::Vector2d> &machineDir,
    double h, double mu, double muPerp);
