#pragma once

#include "shell_mesh.h"
#include "energy.h"
#include "implicit_euler.h"
#include "diffusion.h"
#include "rest_metric.h"
#include "display_mode.h"
#include "graphics/shape.h"
#include <future>
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
    int displayMode() const { return m_shape.displayMode(); }
    int stepCount() const { return m_stepCount; }
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

    // For smooth Hermite interpolation between physics steps.
    std::vector<Eigen::Vector3d> m_prevVertices;
    std::vector<Eigen::Vector3d> m_currVertices;
    std::vector<Eigen::Vector3d> m_prevVelocities;
    std::vector<Eigen::Vector3d> m_currVelocities;
    bool m_hasPhysicsStep = false;
    float m_interpAlpha = 1.0f;

    std::vector<Eigen::Matrix2d> m_a0;
    ShellRestState m_initialRest;

    Shape m_shape;

    double m_dt;
    double m_diffusivity = 0.0;
    std::string m_restMetric;

    bool isSwellingMetric() const {
        return m_restMetric == "swelling_linear" ||
               m_restMetric == "swelling_piecewise" ||
               m_restMetric == "swelling_machine";
    }

    std::vector<double> m_faceEnergies;
    double m_maxInitialEnergy = 0.0;

    // Per-vertex moisture/temperature (top and bottom of shell).
    std::vector<double> m_mPlus;   // top surface
    std::vector<double> m_mMinus;  // bottom surface

    int m_stepCount = 0;
    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;

    // Async physics.
    std::future<void> m_physicsFuture;
    bool m_physicsRunning = false;
    ShellMesh m_physicsMesh;  // physics works on its own copy
    std::vector<Eigen::Vector3d> m_physicsVelocities;

    void waitForPhysics();    // block until in-flight physics finishes
    void launchPhysics();     // start background physics step
    void collectPhysics();    // swap results into main state
};
