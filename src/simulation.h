#pragma once

#include "shell_mesh.h"
#include "energy.h"
#include "implicit_euler.h"
#include "diffusion.h"
#include "rest_metric.h"
#include "display_mode.h"
#include "graphics/shape.h"
#include <chrono>
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
    double avgStepMs() const { return m_avgStepMs; }
    bool stepReady() const { return m_stepReady; }
    void clearStepReady() { m_stepReady = false; }
    void singleStep();
    // Interpolate display between previous and current physics state.
    // alpha: 0 = previous, 1 = current. Call from render loop.
    void interpolateDisplay(float alpha);

    // Moisture painting: raycast from camera and paint moisture at hit point.
    // button: 0 = add m⁺, 1 = add m⁻
    void paintMoisture(const Eigen::Vector3f &rayOrigin,
                       const Eigen::Vector3f &rayDir,
                       int button, float radius = 0.3f, float strength = 0.1f);

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
    std::vector<Eigen::Matrix2d> m_b0;
    ShellRestState m_initialRest;
    double m_swellMu = 0.0025;
    double m_swellMuPerp = 0.001;
    std::vector<Eigen::Vector2d> m_machineDir;

    Shape m_shape;

    double m_dt;
    double m_diffusivity = 0.0;
    std::string m_restMetric;
    std::string m_moistureInit;  // "none" or "one_sided"

    bool isSwellingMetric() const {
        return m_restMetric == "swelling_linear" ||
               m_restMetric == "swelling_piecewise" ||
               m_restMetric == "swelling_machine";
    }

    std::vector<double> m_faceEnergies;
    double m_maxEnergy = 0.0;

    // Per-vertex moisture/temperature (top and bottom of shell).
    std::vector<double> m_mPlus;   // top surface (current, for physics)
    std::vector<double> m_mMinus;  // bottom surface (current, for physics)
    std::vector<double> m_prevMPlus, m_prevMMinus;
    std::vector<double> m_currMPlus, m_currMMinus;

    int m_stepCount = 0;
    double m_avgStepMs = 0.0;
    std::chrono::steady_clock::time_point m_launchTime;
    bool m_stepReady = false;  // true when collectPhysics delivers new results
    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;

    // Async physics.
    std::future<void> m_physicsFuture;
    bool m_physicsRunning = false;
    ShellMesh m_physicsMesh;  // physics works on its own copy
    std::vector<Eigen::Vector3d> m_physicsVelocities;
    std::vector<double> m_physicsMPlus, m_physicsMMinus;
    std::vector<double> m_launchMPlus, m_launchMMinus;  // snapshot at launch for paint merging
    ShellRestState m_physicsRest;  // rest state for background thread

    void waitForPhysics();    // block until in-flight physics finishes
    void launchPhysics();     // start background physics step
    void collectPhysics();    // swap results into main state
};
