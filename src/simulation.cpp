#include "simulation.h"

#include <QSettings>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Eigen;

Simulation::Simulation()
    : m_dt(1e-4),
      m_dampingCoef(5.0),
      m_gravity(0.0, -9.81, 0.0)
{
    omp_set_num_threads(std::max(1, omp_get_num_procs() - 2));
}

void Simulation::init(const std::string &config_path)
{
    m_configPath = config_path;
    QSettings cfg(QString::fromStdString(m_configPath), QSettings::IniFormat);

    cfg.beginGroup("simulation");
    std::string meshPath = cfg.value("mesh").toString().toStdString();
    m_mat.thickness  = cfg.value("thickness",   m_mat.thickness).toDouble();
    m_mat.young      = cfg.value("young",       m_mat.young).toDouble();
    m_mat.poisson    = cfg.value("poisson",     m_mat.poisson).toDouble();
    m_mat.density    = cfg.value("density",     m_mat.density).toDouble();
    m_dt             = cfg.value("dt",          m_dt).toDouble();
    m_dampingCoef    = cfg.value("damping",     m_dampingCoef).toDouble();
    m_growth.rate    = cfg.value("growth_rate", m_growth.rate).toDouble();
    m_gravity.x()    = cfg.value("gravity_x",   m_gravity.x()).toDouble();
    m_gravity.y()    = cfg.value("gravity_y",   m_gravity.y()).toDouble();
    m_gravity.z()    = cfg.value("gravity_z",   m_gravity.z()).toDouble();
    bool runFDCheck  = cfg.value("verify_forces", false).toBool();
    m_restMetric     = cfg.value("rest_metric", "").toString().toStdString();
    cfg.endGroup();

    // ---- Mesh ----
    m_mesh.load(meshPath);
    m_mesh.buildTopology();
    m_shape.init(m_mesh.vertices, m_mesh.faces);

    // ---- Rest state from geometry ----
    const int nF = m_mesh.numFaces();
    m_rest.aBar.resize(nF);
    m_rest.bBar.assign(nF, Matrix2d::Zero());
    m_rest.restArea.resize(nF);
    m_a0.resize(nF);
    for (int f = 0; f < nF; ++f) {
        m_a0[f]          = firstFundamentalForm(m_mesh, f);
        m_rest.aBar[f]   = m_a0[f];
        m_rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, m_a0[f].determinant()));
    }

    // ---- Demo-specific overrides ----
    if (m_restMetric == "stereographic") {
        initStereographicDemo(m_mesh, m_a0, m_rest);
    }

    // ---- Snapshot for reset ----
    m_initialRest  = m_rest;
    m_restVertices = m_mesh.vertices;

    // ---- Per-vertex state ----
    m_velocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_mPlus.assign(m_mesh.numVerts(), 0.0);
    m_mMinus.assign(m_mesh.numVerts(), 0.0);
    computeLumpedMasses(m_mesh, m_rest, m_mat, m_masses);

    // ---- Diagnostics ----
    if (runFDCheck) verifyForceGradient(m_mesh, m_rest, m_mat, m_gravity, m_masses);

    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_mesh.vertices);
}

// =============================================================================
// Time integration
// =============================================================================

void Simulation::stepOnce()
{
    // 1. Growth ramp (swelling demo only).
    if (m_restMetric.empty())
        stepGrowthRamp(m_growth, m_dt, m_a0, m_rest);

    // 2. Implicit Euler with Newton's method (paper Section 5).
    stepImplicitEuler(m_mesh, m_rest, m_mat, m_gravity,
                      m_masses, m_velocities, m_dt);
}

void Simulation::update(double seconds)
{
    if (m_mesh.vertices.empty() || m_paused) return;

    double remaining = seconds;
    while (remaining > 0.0) {
        remaining -= std::min(m_dt, remaining);
        stepOnce();
    }

    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_mesh.vertices);
}

// =============================================================================
// UI / lifecycle
// =============================================================================

void Simulation::togglePause()
{
    m_paused = !m_paused;
    std::cout << (m_paused ? "Paused" : "Running") << std::endl;
}

void Simulation::toggleParallel()
{
    m_parallel = !m_parallel;
    omp_set_num_threads(m_parallel ? std::max(1, omp_get_num_procs() - 2) : 1);
    std::cout << "Parallelization: " << (m_parallel ? "ON" : "OFF") << std::endl;
}

void Simulation::reset()
{
    m_paused = true;
    std::cout << "Reset" << std::endl;
    m_mesh.vertices = m_restVertices;
    std::fill(m_velocities.begin(), m_velocities.end(), Vector3d::Zero());
    std::fill(m_mPlus.begin(),  m_mPlus.end(),  0.0);
    std::fill(m_mMinus.begin(), m_mMinus.end(), 0.0);
    m_growth.factor = 1.0;
    m_growth.target = 1.0;
    m_rest = m_initialRest;
    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_mesh.vertices);
}

void Simulation::setUniformGrowth(double factor)
{
    m_growth.target = factor;
    if (m_paused) {
        m_paused = false;
        std::cout << "Auto-unpaused" << std::endl;
    }
    std::cout << "Growth target = " << factor << std::endl;
}

void Simulation::cycleGrowthDemo()
{
    ::cycleGrowthDemo(m_growth, m_paused);
}

void Simulation::draw(Shader *shader)
{
    m_shape.draw(shader);
}

void Simulation::toggleWire()
{
    m_shape.toggleWireframe();
}
