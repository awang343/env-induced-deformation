#include "simulation.h"
#include "geometry.h"

#include <QSettings>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Eigen;

Simulation::Simulation()
    : m_dt(1e-4)
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
    m_restMetric     = cfg.value("rest_metric", "").toString().toStdString();
    cfg.endGroup();

    m_mesh.load(meshPath);
    m_mesh.buildTopology();
    m_shape.init(m_mesh.vertices, m_mesh.faces);

    const int nF = m_mesh.numFaces();
    m_rest.aBar.resize(nF);
    m_rest.bBar.assign(nF, Matrix2d::Zero());
    m_rest.restArea.resize(nF);
    m_a0.resize(nF);
    for (int f = 0; f < nF; ++f) {
        m_a0[f]            = firstFundamentalForm(m_mesh, f);
        m_rest.aBar[f]     = m_a0[f];
        m_rest.restArea[f] = 0.5 * std::sqrt(std::max(0.0, m_a0[f].determinant()));
    }

    if (!m_restMetric.empty()) {
        cfg.beginGroup("rest_metric");
        int seed       = cfg.value("seed", 42).toInt();
        double perturb = cfg.value("perturb", 0.05).toDouble();

        if (m_restMetric == "sphere_stretching")
            initSphereStretching(m_mesh, m_a0, m_rest, seed, perturb);
        else if (m_restMetric == "sphere_bending")
            initSphereBending(m_mesh, m_a0, m_rest, seed, perturb);
        else if (m_restMetric == "isotropic_growth") {
            double growth = cfg.value("growth_factor", 2.0).toDouble();
            initIsotropicGrowth(m_mesh, m_a0, m_rest, growth, seed, perturb);
        }
        else if (m_restMetric == "cylinder") {
            double kappa = cfg.value("kappa", 1.0).toDouble();
            initCylinderDemo(m_mesh, m_a0, m_rest, kappa, seed, perturb);
        }
        else if (isSwellingMetric()) {
            m_diffusivity = cfg.value("diffusivity", 1e-7).toDouble();
            initSwelling(m_mesh, m_a0, m_rest, seed, perturb);
        }
        else {
            std::cerr << "ERROR: unrecognized rest_metric '" << m_restMetric << "'" << std::endl;
            std::exit(1);
        }

        cfg.endGroup();
    }

    m_initialRest  = m_rest;
    m_restVertices = m_mesh.vertices;

    m_velocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    computeLumpedMasses(m_mesh, m_rest, m_mat, m_masses);

    // Initialize moisture: top=1.0, bottom=0.0 for swelling demos.
    const int nV = m_mesh.numVerts();
    if (isSwellingMetric()) {
        m_mPlus.assign(nV, 1.0);
        m_mMinus.assign(nV, 0.0);
    } else {
        m_mPlus.assign(nV, 0.0);
        m_mMinus.assign(nV, 0.0);
    }

    m_prevVertices = m_mesh.vertices;
    m_currVertices = m_mesh.vertices;
    m_prevVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_currVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_hasPhysicsStep = true;  // allow interpolateDisplay to run from the start

    updateDisplay();

    // Capture max initial energy for heatmap scaling.
    m_maxInitialEnergy = 0.0;
    for (double e : m_faceEnergies)
        m_maxInitialEnergy = std::max(m_maxInitialEnergy, e);
}

void Simulation::stepOnce()
{
    // Diffuse moisture if active.
    if (m_diffusivity > 0.0 && !m_mPlus.empty())
        diffuseMoisture(m_mesh, m_rest, m_mat, m_dt, m_diffusivity,
                        m_mPlus, m_mMinus);

    // TODO: recompute ā, b̄ from m⁺, m⁻ via swelling formulas (Section 2).

    m_prevVertices = m_currVertices;
    m_prevVelocities = m_velocities;
    m_mesh.vertices = m_currVertices;
    stepImplicitEuler(m_mesh, m_rest, m_mat,
                      m_masses, m_velocities, m_dt);
    m_currVertices = m_mesh.vertices;
    m_currVelocities = m_velocities;
    m_interpAlpha = 0.0f;
    m_hasPhysicsStep = true;
}

void Simulation::launchPhysics()
{
    if (m_physicsRunning) return;
    // Copy state for background thread to work on.
    m_physicsMesh.vertices = m_currVertices;
    m_physicsMesh.faces = m_mesh.faces;
    m_physicsMesh.edges = m_mesh.edges;
    m_physicsMesh.faceEdges = m_mesh.faceEdges;
    m_physicsMesh.faceNeighbors = m_mesh.faceNeighbors;
    m_physicsMesh.edgeFaces = m_mesh.edgeFaces;
    m_physicsMesh.vertexFaceOffsets = m_mesh.vertexFaceOffsets;
    m_physicsMesh.vertexFaceList = m_mesh.vertexFaceList;
    m_physicsVelocities = m_velocities;

    // Run diffusion on main thread (fast) so m_mPlus/m_mMinus are
    // safely readable by updateDisplay() without racing the background thread.
    if (m_diffusivity > 0.0 && !m_mPlus.empty())
        diffuseMoisture(m_mesh, m_rest, m_mat, m_dt, m_diffusivity,
                        m_mPlus, m_mMinus);
    // TODO: recompute ā, b̄ from m⁺, m⁻ via swelling formulas (Section 2).

    m_physicsRunning = true;
    m_physicsFuture = std::async(std::launch::async, [this]() {
        stepImplicitEuler(m_physicsMesh, m_rest, m_mat,
                          m_masses, m_physicsVelocities, m_dt);
    });
}

void Simulation::collectPhysics()
{
    if (!m_physicsRunning) return;
    m_physicsFuture.get();
    m_physicsRunning = false;
    m_stepCount++;

    m_prevVertices = m_currVertices;
    m_prevVelocities = m_currVelocities;
    m_currVertices = m_physicsMesh.vertices;
    m_currVelocities = m_physicsVelocities;
    m_velocities = m_physicsVelocities;
    m_interpAlpha = 0.0f;
    m_hasPhysicsStep = true;
}

void Simulation::waitForPhysics()
{
    if (m_physicsRunning) {
        m_physicsFuture.get();
        m_physicsRunning = false;
    }
}

void Simulation::update(double /*seconds*/)
{
    if (m_mesh.vertices.empty() || m_paused) return;

    // Check if background physics finished.
    if (m_physicsRunning &&
        m_physicsFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        collectPhysics();
    }

    // Launch next step if not already running.
    if (!m_physicsRunning)
        launchPhysics();
}

void Simulation::interpolateDisplay(float alpha)
{
    if (!m_hasPhysicsStep) return;
    const double t = std::min(1.0, std::max(0.0, (double)alpha));
    const int n = m_mesh.numVerts();

    // Cubic Hermite interpolation using positions and velocities at
    // both endpoints. Gives C1 continuity (smooth velocity) across
    // physics step boundaries.
    //   p(t) = h00·p0 + h10·dt·v0 + h01·p1 + h11·dt·v1
    // where h00 = 2t³-3t²+1, h10 = t³-2t²+t, h01 = -2t³+3t², h11 = t³-t²
    const double t2 = t * t, t3 = t2 * t;
    const double h00 = 2*t3 - 3*t2 + 1;
    const double h10 = t3 - 2*t2 + t;
    const double h01 = -2*t3 + 3*t2;
    const double h11 = t3 - t2;

    for (int i = 0; i < n; ++i)
        m_mesh.vertices[i] = h00 * m_prevVertices[i]
                            + h10 * m_dt * m_prevVelocities[i]
                            + h01 * m_currVertices[i]
                            + h11 * m_dt * m_currVelocities[i];
    updateDisplay();
}

void Simulation::updateDisplay()
{
    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_mesh.vertices);

    const int nF = m_mesh.numFaces();
    double alpha = m_mat.alpha(), beta = m_mat.beta();
    double h3_12 = m_mat.thickness*m_mat.thickness*m_mat.thickness / 12.0;
    std::vector<Eigen::Vector3d> fN;
    computeFaceNormals(m_mesh, fN);
    m_faceEnergies.resize(nF);
    for (int f = 0; f < nF; ++f) {
        Matrix2d a = firstFundamentalForm(m_mesh, f);
        Matrix2d aBI = m_rest.aBar[f].inverse();
        Matrix2d Ms = aBI * a - Matrix2d::Identity();
        double es = 0.25 * m_mat.thickness * m_rest.restArea[f]
                  * (0.5*alpha*Ms.trace()*Ms.trace() + beta*(Ms*Ms).trace());
        Matrix2d b = secondFundamentalForm(m_mesh, fN, f);
        Matrix2d Mb = aBI * (b - m_rest.bBar[f]);
        double eb = h3_12 * m_rest.restArea[f]
                  * (0.5*alpha*Mb.trace()*Mb.trace() + beta*(Mb*Mb).trace());
        m_faceEnergies[f] = es + eb;
    }
    // Channel 0: log-normalized energy (for energy heatmap + solid/wireframe).
    std::vector<double> ch0(nF, 0.0);
    if (m_maxInitialEnergy > 0.0) {
        double logMax = std::log(1.0 + m_maxInitialEnergy);
        for (int f = 0; f < nF; ++f)
            ch0[f] = std::log(1.0 + m_faceEnergies[f]) / logMax;
    }

    // Channel 1: m_minus per face (for moisture back-face display).
    // Channel 0 doubles as m_plus when in moisture mode.
    std::vector<double> ch1(nF, 0.0);

    int mode = m_shape.displayMode();
    if (mode == 2 && !m_mPlus.empty()) {
        // Override ch0 with m_plus, ch1 with m_minus for moisture display.
        for (int f = 0; f < nF; ++f) {
            const auto &tri = m_mesh.faces[f];
            ch0[f] = (m_mPlus[tri[0]] + m_mPlus[tri[1]] + m_mPlus[tri[2]]) / 3.0;
            ch1[f] = (m_mMinus[tri[0]] + m_mMinus[tri[1]] + m_mMinus[tri[2]]) / 3.0;
        }
    }

    m_shape.setFaceData(ch0, ch1);
}

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
    waitForPhysics();
    m_paused = true;
    std::cout << "Reset" << std::endl;
    m_mesh.vertices = m_restVertices;
    std::fill(m_velocities.begin(), m_velocities.end(), Vector3d::Zero());
    m_rest = m_initialRest;
    m_prevVertices = m_restVertices;
    m_currVertices = m_restVertices;
    m_prevVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_currVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_stepCount = 0;
    m_hasPhysicsStep = true;
    if (isSwellingMetric()) {
        m_mPlus.assign(m_mesh.numVerts(), 1.0);
        m_mMinus.assign(m_mesh.numVerts(), 0.0);
    }
    updateDisplay();
}

void Simulation::singleStep()
{
    waitForPhysics();
    stepOnce();
    m_stepCount++;
    m_mesh.vertices = m_currVertices;
    m_prevVertices = m_currVertices;
    updateDisplay();
    std::cout << "Step" << std::endl;
}

void Simulation::draw(Shader *shader) { m_shape.draw(shader); }
void Simulation::toggleWire()
{
    if (isSwellingMetric()) {
        m_shape.cycleDisplayMode(4);
    } else {
        int mode = m_shape.displayMode();
        if (mode == 0) m_shape.cycleDisplayMode(4);
        else if (mode == 1) { m_shape.cycleDisplayMode(4);
                              m_shape.cycleDisplayMode(4); }
        else m_shape.cycleDisplayMode(4);
    }
    // Refresh VBO with correct data for the new mode.
    m_mesh.vertices = m_currVertices;
    updateDisplay();
}
