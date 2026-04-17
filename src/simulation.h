#pragma once

#include "shell_mesh.h"
#include "shell_energy.h"
#include "demo.h"
#include "graphics/shape.h"
#include <string>

class Shader;

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
    bool isPaused() const { return m_paused; }
    void setUniformGrowth(double factor);
    void cycleGrowthDemo();
    void singleStep();

  private:
    void stepOnce();
    void updateDisplay();

    ShellMesh        m_mesh;
    ShellRestState   m_rest;
    MaterialParams   m_mat;
    DampingState     m_damp;

    std::vector<Eigen::Vector3d> m_restVertices;
    std::vector<Eigen::Vector3d> m_velocities;
    std::vector<double>          m_masses;

    std::vector<Eigen::Matrix2d> m_a0;
    ShellRestState m_initialRest;

    Shape m_shape;

    double m_dt;
    Eigen::Vector3d m_gravity;

    GrowthState m_growth;
    std::string m_restMetric;

    std::vector<double> m_faceEnergies;  // cached for heatmap upload

    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;
};
