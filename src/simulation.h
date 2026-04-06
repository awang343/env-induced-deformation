#pragma once

#include "graphics/shape.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

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

  private:
    Shape m_shape;
    Shape m_ground;
    Shape m_sphere;

    void loadMesh(const std::string &meshPath);
    void initGround();
    void initSphere();

    // Simulation state
    std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector3d> m_restVertices;
    std::vector<Eigen::Vector3d> m_velocities;
    std::vector<Eigen::Vector4i> m_tets;

    // Precomputed per-tet data
    std::vector<Eigen::Matrix3d> m_DmInv;
    std::vector<double> m_restVolumes;

    // Per-vertex masses (lumped mass model)
    std::vector<double> m_masses;

    // Material parameters (loaded from config)
    double m_lambda;
    double m_mu;
    double m_phi;
    double m_psi;
    double m_density;
    double m_dt;
    Eigen::Vector3d m_gravity;

    // Collision parameters
    double m_groundY;
    double m_restitution;
    double m_friction;

    // Sphere obstacle
    bool m_sphereEnabled;
    double m_sphereRadius;
    Eigen::Vector3d m_sphereCenter;
    std::string m_sphereMeshPath;

    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;

    // Precomputed per-tet combined matrices
    std::vector<Eigen::Matrix3d> m_DmInvT;         // Dm^{-T}
    std::vector<Eigen::Matrix3d> m_volDmInvT;       // V * Dm^{-T}
    std::vector<double> m_invMasses;                 // 1/mass per vertex

    // Pre-allocated scratch buffers (avoid heap allocs per substep)
    std::vector<Eigen::Vector3d> m_forces;
    std::vector<Eigen::Vector3d> m_k1x, m_k1v;
    std::vector<Eigen::Vector3d> m_k2x, m_k2v;
    std::vector<Eigen::Vector3d> m_k3x, m_k3v;
    std::vector<Eigen::Vector3d> m_k4x, m_k4v;
    std::vector<Eigen::Vector3d> m_tmpPos, m_tmpVel;
    std::vector<std::vector<Eigen::Vector3d>> m_threadForces;

    // Core methods
    void precompute();
    void allocateScratch();
    void computeForces(const std::vector<Eigen::Vector3d> &positions,
                       const std::vector<Eigen::Vector3d> &velocities,
                       std::vector<Eigen::Vector3d> &forces);
    void resolveCollisions();
};
