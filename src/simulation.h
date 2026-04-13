#pragma once

#include "shell_mesh.h"
#include "shell_energy.h"
#include "demo.h"
#include "graphics/shape.h"
#include <string>

class Shader;

// Orchestrator: owns the mesh, rest state, material, and rendering.
// Delegates energy/force computation to shell_energy and demo logic to demo.
class Simulation
{
  public:
    Simulation();

    void init(const std::string &config_path);
    void update(double seconds);
    void draw(Shader *shader);

    void toggleWire();
    void togglePause();
    void toggleParallel();
    void reset();

    void setUniformGrowth(double factor);
    void cycleGrowthDemo();

  private:
    void stepOnce();

    // ---- Core state ----
    ShellMesh      m_mesh;
    ShellRestState m_rest;
    MaterialParams m_mat;

    std::vector<Eigen::Vector3d> m_restVertices;
    std::vector<Eigen::Vector3d> m_velocities;
    std::vector<double>          m_masses;
    std::vector<double>          m_mPlus, m_mMinus;

    // Reference rest forms (for computing growth / reset).
    std::vector<Eigen::Matrix2d> m_a0;
    // Snapshot of aBar/restArea at init (for reset).
    ShellRestState m_initialRest;

    // ---- Rendering ----
    Shape m_shape;

    // ---- Integration parameters ----
    double m_dt;
    double m_dampingCoef;
    Eigen::Vector3d m_gravity;

    // ---- Demo state ----
    GrowthState m_growth;
    std::string m_restMetric;

    // ---- Flags ----
    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;
};
