#pragma once

#include "shell_mesh.h"
#include "shell_energy.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

// Demo-specific setup and runtime logic for the checkpoint demos.
// None of this is part of the paper's formulation — it's scaffolding
// for driving the simulation in interesting ways with explicit Euler.

// ---- Uniform growth (swelling square demo, Table 2 row 1) ----

struct GrowthState
{
    double factor = 1.0;   // current factor applied to a0
    double target = 1.0;   // what cycleGrowthDemo() asks for
    double rate   = 0.3;   // factor units per second
};

// Set aBar = factor^2 * a0 for every face.
void applyGrowthFactor(double factor,
                       const std::vector<Eigen::Matrix2d> &a0,
                       ShellRestState &rest);

// Advance factor toward target by rate*dt. Returns true if rest state changed.
bool stepGrowthRamp(GrowthState &gs, double dt,
                    const std::vector<Eigen::Matrix2d> &a0,
                    ShellRestState &rest);

// Bump the target up/down for keyboard demos.
void cycleGrowthDemo(GrowthState &gs, bool &paused);

// ---- Stereographic disk-to-sphere (Figure 3) ----

// Compute the stereographic rest metric for a flat disk mesh centered at
// the origin in the xz plane. Sets aBar = (2/(1+r²))² * a0 per face.
// Also repositions vertices onto a partial hemisphere (initBlend in [0,1])
// so explicit Euler starts near the equilibrium.
void initStereographicDemo(ShellMesh &mesh,
                           const std::vector<Eigen::Matrix2d> &a0,
                           ShellRestState &rest,
                           int seed = 42,
                           double perturbScale = 0.05);
