#include "simulation.h"

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
    m_growth.rate    = cfg.value("growth_rate", m_growth.rate).toDouble();
    m_restMetric     = cfg.value("rest_metric", "").toString().toStdString();
    int seed         = cfg.value("seed", 42).toInt();
    double perturb   = cfg.value("perturb", 0.05).toDouble();
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

    if (m_restMetric == "stereographic")
        initStereographicDemo(m_mesh, m_a0, m_rest, seed, perturb);

    m_initialRest  = m_rest;
    m_restVertices = m_mesh.vertices;

    m_velocities.assign(m_mesh.numVerts(), Vector3d::Zero());
    computeLumpedMasses(m_mesh, m_rest, m_mat, m_masses);

    updateDisplay();
}

void Simulation::stepOnce()
{
    if (m_restMetric.empty())
        stepGrowthRamp(m_growth, m_dt, m_a0, m_rest);

    stepImplicitEuler(m_mesh, m_rest, m_mat,
                      m_masses, m_velocities, m_dt);
}

void Simulation::update(double /*seconds*/)
{
    if (m_mesh.vertices.empty() || m_paused) return;
    stepOnce();
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
    m_shape.setFaceEnergies(m_faceEnergies);
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
    m_paused = true;
    std::cout << "Reset" << std::endl;
    m_mesh.vertices = m_restVertices;
    std::fill(m_velocities.begin(), m_velocities.end(), Vector3d::Zero());
    m_growth.factor = 1.0;
    m_growth.target = 1.0;
    m_rest = m_initialRest;
    updateDisplay();
}

void Simulation::setUniformGrowth(double factor)
{
    m_growth.target = factor;
    if (m_paused) { m_paused = false; std::cout << "Auto-unpaused" << std::endl; }
    std::cout << "Growth target = " << factor << std::endl;
}

void Simulation::cycleGrowthDemo() { ::cycleGrowthDemo(m_growth, m_paused); }

void Simulation::singleStep()
{
    stepOnce();
    updateDisplay();
    std::cout << "Step" << std::endl;
}

void Simulation::draw(Shader *shader) { m_shape.draw(shader); }
void Simulation::toggleWire() { m_shape.cycleDisplayMode(); }
