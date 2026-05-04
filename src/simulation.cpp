#include "simulation.h"
#include "geometry.h"
#include "simplify/simplify_mesh.h"

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
    m_moistureInit   = cfg.value("moisture",    "none").toString().toStdString();
    m_diffusivity    = cfg.value("diffusivity", 0.0).toDouble();
    m_simplification = cfg.value("simplification", 0.0).toDouble();
    cfg.endGroup();

    // Load display (high-res) mesh.
    m_displayMesh.load(meshPath);
    m_displayMesh.buildTopology();
    m_displayShape.init(m_displayMesh.vertices, m_displayMesh.faces);
    m_displayVertices = m_displayMesh.vertices;

    // Create physics mesh (simplified or same as display).
    if (m_simplification > 0.0) {
        std::vector<Vector3d> simVerts;
        std::vector<Vector3i> simFaces;
        simplifyMesh(m_displayMesh.vertices, m_displayMesh.faces,
                     m_simplification, simVerts, simFaces);
        m_mesh.vertices = simVerts;
        m_mesh.faces = simFaces;
        m_mesh.buildTopology();
        computeEmbedding(m_displayMesh, m_mesh, m_displayEmbed);
    } else {
        m_mesh = m_displayMesh;
    }
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
        else if (m_restMetric == "sphere_target") {
            double radius = cfg.value("radius", 1.0).toDouble();
            initSphereTarget(m_mesh, m_a0, m_rest, radius, seed, perturb);
        }
        else if (m_restMetric == "cylinder") {
            double kappa = cfg.value("kappa", 1.0).toDouble();
            initCylinderDemo(m_mesh, m_a0, m_rest, kappa, seed, perturb);
        }
        else if (isSwellingMetric()) {
            m_swellMu     = cfg.value("mu", 0.0025).toDouble();
            m_swellMuPerp = cfg.value("mu_perp", 0.001).toDouble();
            initSwelling(m_mesh, m_a0, m_rest, seed, perturb);
        }
        else {
            std::cerr << "ERROR: unrecognized rest_metric '" << m_restMetric << "'" << std::endl;
            std::exit(1);
        }

        cfg.endGroup();
    }

    m_initialRest  = m_rest;
    m_b0 = m_rest.bBar;  // store initial b for swelling updates
    m_restVertices = m_mesh.vertices;

    m_velocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    computeLumpedMasses(m_mesh, m_rest, m_mat, m_masses);

    // Initialize moisture.
    const int nV = m_mesh.numVerts();
    if (m_moistureInit == "one_sided") {
        m_mPlus.assign(nV, 1.0);
        m_mMinus.assign(nV, 0.0);
    } else {
        m_mPlus.assign(nV, 0.0);
        m_mMinus.assign(nV, 0.0);
    }

    m_prevMPlus = m_mPlus;  m_currMPlus = m_mPlus;
    m_prevMMinus = m_mMinus; m_currMMinus = m_mMinus;

    m_prevVertices = m_mesh.vertices;
    m_currVertices = m_mesh.vertices;
    m_prevVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_currVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_hasPhysicsStep = true;  // allow interpolateDisplay to run from the start

    updateDisplay();

    m_maxEnergy = 0.0;
    for (double e : m_faceEnergies)
        m_maxEnergy = std::max(m_maxEnergy, e);
}

void Simulation::stepOnce()
{
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
    m_physicsMPlus = m_mPlus;
    m_physicsMMinus = m_mMinus;
    m_launchMPlus = m_mPlus;
    m_launchMMinus = m_mMinus;
    m_physicsRest = m_rest;

    m_physicsRunning = true;
    m_launchTime = std::chrono::steady_clock::now();
    m_physicsFuture = std::async(std::launch::async, [this]() {
        // Diffuse moisture.
        if (m_diffusivity > 0.0)
            diffuseMoisture(m_physicsMesh, m_physicsRest, m_mat, m_dt,
                            m_diffusivity, m_physicsMPlus, m_physicsMMinus);

        // Update rest forms from moisture.
        if (m_restMetric == "swelling_linear")
            updateRestFormsLinear(m_physicsMesh, m_physicsRest, m_a0, m_b0,
                                 m_physicsMPlus, m_physicsMMinus, m_mat.thickness, m_swellMu);
        else if (m_restMetric == "swelling_piecewise")
            updateRestFormsPiecewise(m_physicsMesh, m_physicsRest, m_a0, m_b0,
                                    m_physicsMPlus, m_physicsMMinus, m_mat.thickness, m_swellMu);
        else if (m_restMetric == "swelling_machine")
            updateRestFormsMachine(m_physicsMesh, m_physicsRest, m_a0, m_b0,
                                  m_physicsMPlus, m_physicsMMinus, m_machineDir,
                                  m_mat.thickness, m_swellMu, m_swellMuPerp);

        // Solve mechanics with updated rest forms.
        stepImplicitEuler(m_physicsMesh, m_physicsRest, m_mat,
                          m_masses, m_physicsVelocities, m_dt);
    });
}

void Simulation::collectPhysics()
{
    if (!m_physicsRunning) return;
    m_physicsFuture.get();
    m_physicsRunning = false;
    m_stepCount++;
    m_avgStepMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - m_launchTime).count();

    // Copy all results from background thread.
    m_prevVertices = m_currVertices;
    m_prevVelocities = m_currVelocities;
    m_prevMPlus = m_currMPlus;
    m_prevMMinus = m_currMMinus;
    m_currVertices = m_physicsMesh.vertices;
    m_currVelocities = m_physicsVelocities;
    // Merge any moisture painted during this step on top of physics result.
    const int nV = m_mesh.numVerts();
    for (int i = 0; i < nV; ++i) {
        double paintDeltaP = m_mPlus[i] - m_launchMPlus[i];
        double paintDeltaM = m_mMinus[i] - m_launchMMinus[i];
        m_mPlus[i]  = std::clamp(m_physicsMPlus[i] + paintDeltaP, 0.0, 1.0);
        m_mMinus[i] = std::clamp(m_physicsMMinus[i] + paintDeltaM, 0.0, 1.0);
    }
    m_currMPlus = m_mPlus;
    m_currMMinus = m_mMinus;
    m_rest = m_physicsRest;

    // Re-apply rest form update from current moisture (which may include
    // paint applied while this step was in flight).
    if (m_restMetric == "swelling_linear")
        updateRestFormsLinear(m_mesh, m_rest, m_a0, m_b0,
                             m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
    else if (m_restMetric == "swelling_piecewise")
        updateRestFormsPiecewise(m_mesh, m_rest, m_a0, m_b0,
                                m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
    else if (m_restMetric == "swelling_machine")
        updateRestFormsMachine(m_mesh, m_rest, m_a0, m_b0,
                              m_mPlus, m_mMinus, m_machineDir,
                              m_mat.thickness, m_swellMu, m_swellMuPerp);

    m_velocities = m_physicsVelocities;
    m_interpAlpha = 0.0f;
    m_hasPhysicsStep = true;
    m_stepReady = true;
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

    // Linear interpolation for moisture.
    for (int i = 0; i < n; ++i) {
        m_mPlus[i]  = (1.0 - t) * m_prevMPlus[i]  + t * m_currMPlus[i];
        m_mMinus[i] = (1.0 - t) * m_prevMMinus[i] + t * m_currMMinus[i];
    }

    updateDisplay();
}

void Simulation::transferDeformation()
{
    if (m_simplification <= 0.0) return;
    const int nD = m_displayMesh.numVerts();
    m_displayVertices.resize(nD);
    for (int i = 0; i < nD; ++i) {
        const auto &emb = m_displayEmbed[i];
        const auto &tri = m_mesh.faces[emb.face];
        m_displayVertices[i] = emb.bary[0] * m_mesh.vertices[tri[0]]
                             + emb.bary[1] * m_mesh.vertices[tri[1]]
                             + emb.bary[2] * m_mesh.vertices[tri[2]];
    }
}

void Simulation::updateDisplay()
{
    int mode = m_shape.displayMode();

    // Solid mode with simplification: show high-res display mesh.
    if (mode == 0 && m_simplification > 0.0) {
        transferDeformation();
        m_displayShape.setModelMatrix(Affine3f::Identity());
        m_displayShape.setVertices(m_displayVertices);
    }

    // Always update physics mesh shape (for energy/moisture/wireframe modes,
    // and to keep face energies current for the overlay).
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
    std::vector<double> ch0(nF, 0.0);
    if (m_maxEnergy > 0.0) {
        double logMax = std::log(1.0 + m_maxEnergy);
        for (int f = 0; f < nF; ++f)
            ch0[f] = std::log(1.0 + m_faceEnergies[f]) / logMax;
    }

    std::vector<double> ch1(nF, 0.0);
    if (mode == 2) {
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
    if (!m_paused) {
        // Update rest forms from current moisture before rescaling.
        if (m_restMetric == "swelling_linear")
            updateRestFormsLinear(m_mesh, m_rest, m_a0, m_b0,
                                 m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
        else if (m_restMetric == "swelling_piecewise")
            updateRestFormsPiecewise(m_mesh, m_rest, m_a0, m_b0,
                                    m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
        else if (m_restMetric == "swelling_machine")
            updateRestFormsMachine(m_mesh, m_rest, m_a0, m_b0,
                                  m_mPlus, m_mMinus, m_machineDir,
                                  m_mat.thickness, m_swellMu, m_swellMuPerp);
        updateDisplay();
        // Rescale energy display to current max.
        m_maxEnergy = 0.0;
        for (double e : m_faceEnergies)
            m_maxEnergy = std::max(m_maxEnergy, e);
    }
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
    m_b0 = m_initialRest.bBar;
    m_prevVertices = m_restVertices;
    m_currVertices = m_restVertices;
    m_prevVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_currVelocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    m_stepCount = 0;
    m_maxEnergy = 0.0;
    m_hasPhysicsStep = true;
    if (m_moistureInit == "one_sided") {
        m_mPlus.assign(m_mesh.numVerts(), 1.0);
        m_mMinus.assign(m_mesh.numVerts(), 0.0);
    } else {
        m_mPlus.assign(m_mesh.numVerts(), 0.0);
        m_mMinus.assign(m_mesh.numVerts(), 0.0);
    }
    m_prevMPlus = m_mPlus;  m_currMPlus = m_mPlus;
    m_prevMMinus = m_mMinus; m_currMMinus = m_mMinus;
    m_displayVertices = m_displayMesh.vertices;
    updateDisplay();
}

void Simulation::singleStep()
{
    waitForPhysics();
    if (m_diffusivity > 0.0)
        diffuseMoisture(m_mesh, m_rest, m_mat, m_dt, m_diffusivity,
                        m_mPlus, m_mMinus);
    if (m_restMetric == "swelling_linear")
        updateRestFormsLinear(m_mesh, m_rest, m_a0, m_b0,
                             m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
    else if (m_restMetric == "swelling_piecewise")
        updateRestFormsPiecewise(m_mesh, m_rest, m_a0, m_b0,
                                m_mPlus, m_mMinus, m_mat.thickness, m_swellMu);
    else if (m_restMetric == "swelling_machine")
        updateRestFormsMachine(m_mesh, m_rest, m_a0, m_b0,
                              m_mPlus, m_mMinus, m_machineDir,
                              m_mat.thickness, m_swellMu, m_swellMuPerp);
    stepOnce();
    m_stepCount++;
    m_currMPlus = m_mPlus;
    m_currMMinus = m_mMinus;
    m_mesh.vertices = m_currVertices;
    m_prevVertices = m_currVertices;
    m_prevMPlus = m_currMPlus;
    m_prevMMinus = m_currMMinus;
    updateDisplay();
    std::cout << "Step" << std::endl;
}

void Simulation::paintMoisture(const Eigen::Vector3f &rayOrigin,
                               const Eigen::Vector3f &rayDir,
                               int /*button*/, float radius, float strength)
{
    // Raycast against whichever mesh the user sees.
    int mode = m_shape.displayMode();
    bool useDisplay = (mode == 0 && m_simplification > 0.0);
    const auto &rayVerts = useDisplay ? m_displayVertices : m_mesh.vertices;
    const auto &rayFaces = useDisplay ? m_displayMesh.faces : m_mesh.faces;
    int numRayFaces = (int)rayFaces.size();

    float bestT = std::numeric_limits<float>::max();
    int hitFace = -1;
    Eigen::Vector3f hitPoint;

    for (int f = 0; f < numRayFaces; ++f) {
        const auto &tri = rayFaces[f];
        Eigen::Vector3f v0 = rayVerts[tri[0]].cast<float>();
        Eigen::Vector3f v1 = rayVerts[tri[1]].cast<float>();
        Eigen::Vector3f v2 = rayVerts[tri[2]].cast<float>();

        Eigen::Vector3f e1 = v1 - v0, e2 = v2 - v0;
        Eigen::Vector3f h = rayDir.cross(e2);
        float a = e1.dot(h);
        if (std::abs(a) < 1e-8f) continue;

        float invA = 1.0f / a;
        Eigen::Vector3f s = rayOrigin - v0;
        float u = invA * s.dot(h);
        if (u < 0.0f || u > 1.0f) continue;

        Eigen::Vector3f q = s.cross(e1);
        float v = invA * rayDir.dot(q);
        if (v < 0.0f || u + v > 1.0f) continue;

        float t = invA * e2.dot(q);
        if (t > 1e-4f && t < bestT) {
            bestT = t;
            hitFace = f;
            hitPoint = rayOrigin + t * rayDir;
        }
    }

    if (hitFace < 0) return;

    // Determine which side was clicked: front face → m⁺, back face → m⁻.
    const auto &hitTri = rayFaces[hitFace];
    Eigen::Vector3f faceNormal = (rayVerts[hitTri[1]] - rayVerts[hitTri[0]])
                                  .cross(rayVerts[hitTri[2]] - rayVerts[hitTri[0]])
                                  .cast<float>();
    bool frontFace = rayDir.dot(faceNormal) < 0;

    // Paint moisture on nearby vertices within radius.
    Eigen::Vector3d hp = hitPoint.cast<double>();
    for (int i = 0; i < m_mesh.numVerts(); ++i) {
        double dist = (m_mesh.vertices[i] - hp).norm();
        if (dist < radius) {
            double falloff = 1.0 - dist / radius;
            double delta = strength * falloff;
            if (frontFace)
                m_mPlus[i] = std::min(1.0, m_mPlus[i] + delta);
            else
                m_mMinus[i] = std::min(1.0, m_mMinus[i] + delta);
        }
    }
    m_currMPlus = m_mPlus;
    m_currMMinus = m_mMinus;
    m_prevMPlus = m_mPlus;
    m_prevMMinus = m_mMinus;
    updateDisplay();
}

void Simulation::draw(Shader *shader)
{
    int mode = m_shape.displayMode();
    if (mode == 0 && m_simplification > 0.0)
        m_displayShape.draw(shader);
    else
        m_shape.draw(shader);
}
void Simulation::toggleWire()
{
    m_shape.cycleDisplayMode(4);
    m_displayShape.cycleDisplayMode(4);
    // Refresh VBO with correct data for the new mode.
    m_mesh.vertices = m_currVertices;
    updateDisplay();
}
