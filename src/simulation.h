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
    void singleStep();
    // Interpolate display between previous and current physics state.
    // alpha: 0 = previous, 1 = current. Call from render loop.
    void interpolateDisplay(float alpha);

  private:
    void stepOnce();
    void updateDisplay();

    ShellMesh        m_mesh;
    ShellRestState   m_rest;
    MaterialParams   m_mat;

    std::vector<Eigen::Vector3d> m_restVertices;
    std::vector<Eigen::Vector3d> m_velocities;
    std::vector<double>          m_masses;

    // For smooth interpolation between physics steps.
    std::vector<Eigen::Vector3d> m_prevVertices;
    std::vector<Eigen::Vector3d> m_currVertices;
    bool m_hasPhysicsStep = false;
    float m_interpAlpha = 1.0f;  // 0 = prevVertices, 1 = currVertices

    std::vector<Eigen::Matrix2d> m_a0;
    ShellRestState m_initialRest;

    Shape m_shape;

    double m_dt;
    std::string m_restMetric;

    std::vector<double> m_faceEnergies;
    double m_maxInitialEnergy = 0.0;

    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;
};
