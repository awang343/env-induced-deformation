#include "simulation.h"
#include "graphics/meshloader.h"

#include <QSettings>

#include <omp.h>


#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace Eigen;

Simulation::Simulation()
    : m_lambda(4e3), m_mu(4e3), m_phi(100), m_psi(100),
      m_density(1200), m_dt(0.0003),
      m_gravity(0, -1, 0),
      m_groundY(0), m_restitution(0.2), m_friction(0.9),
      m_sphereEnabled(false), m_sphereRadius(0.5),
      m_sphereCenter(0, 0, 0)
{
    omp_set_num_threads(std::max(1, omp_get_num_procs() - 2));
}

void Simulation::init(const std::string &config_path)
{
    m_configPath = config_path;
    QSettings cfg(QString::fromStdString(m_configPath), QSettings::IniFormat);

    // Load simulation parameters
    cfg.beginGroup("simulation");
    std::string meshPath = cfg.value("mesh").toString().toStdString();
    m_lambda     = cfg.value("lambda", m_lambda).toDouble();
    m_mu         = cfg.value("mu", m_mu).toDouble();
    m_phi        = cfg.value("phi", m_phi).toDouble();
    m_psi        = cfg.value("psi", m_psi).toDouble();
    m_density    = cfg.value("density", m_density).toDouble();
    m_dt         = cfg.value("dt", m_dt).toDouble();
    m_gravity.x() = cfg.value("gravity_x", m_gravity.x()).toDouble();
    m_gravity.y() = cfg.value("gravity_y", m_gravity.y()).toDouble();
    m_gravity.z() = cfg.value("gravity_z", m_gravity.z()).toDouble();
    m_restitution = cfg.value("restitution", m_restitution).toDouble();
    m_friction    = cfg.value("friction", m_friction).toDouble();
    cfg.endGroup();

    // Load sphere obstacle config
    cfg.beginGroup("sphere_obstacle");
    m_sphereEnabled = cfg.value("enabled", false).toBool();
    if (m_sphereEnabled) {
        m_sphereMeshPath = cfg.value("obstacle_mesh").toString().toStdString();
        m_sphereCenter.x() = cfg.value("center_x", 0).toDouble();
        m_sphereCenter.y() = cfg.value("center_y", 0).toDouble();
        m_sphereCenter.z() = cfg.value("center_z", 0).toDouble();
        m_sphereRadius     = cfg.value("radius", 0.5).toDouble();
    }
    cfg.endGroup();

    loadMesh(meshPath);

    // Offset rest and current vertices by the initial translation
    for (auto &v : m_vertices) {
        v.y() += 2.0;
    }
    m_restVertices = m_vertices;

    // Push offset vertices to GPU immediately
    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_vertices);

    // Initialize velocities to zero
    m_velocities.assign(m_vertices.size(), Vector3d::Zero());

    // Precompute per-tet inverse rest shape matrices, volumes, and masses
    precompute();

    initGround();
    if (m_sphereEnabled) {
        initSphere();
    }
}

void Simulation::precompute()
{
    const int numTets = m_tets.size();
    const int numVerts = m_vertices.size();

    m_DmInv.resize(numTets);
    m_DmInvT.resize(numTets);
    m_volDmInvT.resize(numTets);
    m_restVolumes.resize(numTets);
    m_masses.assign(numVerts, 0.0);

    for (int t = 0; t < numTets; ++t) {
        const Vector4i &tet = m_tets[t];
        const Vector3d &x0 = m_restVertices[tet[0]];
        const Vector3d &x1 = m_restVertices[tet[1]];
        const Vector3d &x2 = m_restVertices[tet[2]];
        const Vector3d &x3 = m_restVertices[tet[3]];

        Matrix3d Dm;
        Dm.col(0) = x1 - x0;
        Dm.col(1) = x2 - x0;
        Dm.col(2) = x3 - x0;

        m_DmInv[t] = Dm.inverse();
        m_DmInvT[t] = m_DmInv[t].transpose();

        double vol = std::abs(Dm.determinant()) / 6.0;
        m_restVolumes[t] = vol;
        m_volDmInvT[t] = vol * m_DmInvT[t];

        double elemMass = m_density * vol;
        for (int i = 0; i < 4; ++i) {
            m_masses[tet[i]] += elemMass / 4.0;
        }
    }

    // Precompute inverse masses
    m_invMasses.resize(numVerts);
    for (int i = 0; i < numVerts; ++i) {
        m_invMasses[i] = 1.0 / m_masses[i];
    }

    allocateScratch();
}

void Simulation::allocateScratch()
{
    const int n = m_vertices.size();
    const int numVerts = n;

    m_forces.resize(n);
    m_k1x.resize(n); m_k1v.resize(n);
    m_k2x.resize(n); m_k2v.resize(n);
    m_k3x.resize(n); m_k3v.resize(n);
    m_k4x.resize(n); m_k4v.resize(n);
    m_tmpPos.resize(n); m_tmpVel.resize(n);

    const int numThreads = omp_get_max_threads();
    m_threadForces.resize(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        m_threadForces[t].resize(numVerts);
    }
}

void Simulation::computeForces(const std::vector<Vector3d> &positions,
                                const std::vector<Vector3d> &velocities,
                                std::vector<Vector3d> &forces)
{
    const int numVerts = positions.size();
    const int numTets = m_tets.size();
    const int numThreads = omp_get_max_threads();
    const Matrix3d I = Matrix3d::Identity();

    // Zero gravity + thread buffers in one parallel region
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < numVerts; ++i) {
            forces[i] = m_masses[i] * m_gravity;
        }
        // Zero only this thread's buffer
        std::fill(m_threadForces[tid].begin(), m_threadForces[tid].end(), Vector3d::Zero());

        #pragma omp for schedule(static)
        for (int t = 0; t < numTets; ++t) {
            const Vector4i &tet = m_tets[t];
            const Vector3d &p0 = positions[tet[0]];

            Matrix3d Ds;
            Ds.col(0) = positions[tet[1]] - p0;
            Ds.col(1) = positions[tet[2]] - p0;
            Ds.col(2) = positions[tet[3]] - p0;

            const Matrix3d &DmI = m_DmInv[t];
            Matrix3d F = Ds * DmI;
            Matrix3d FtF = F.transpose() * F;
            Matrix3d E = 0.5 * (FtF - I);

            const Vector3d &v0 = velocities[tet[0]];
            Matrix3d Ds_dot;
            Ds_dot.col(0) = velocities[tet[1]] - v0;
            Ds_dot.col(1) = velocities[tet[2]] - v0;
            Ds_dot.col(2) = velocities[tet[3]] - v0;

            Matrix3d F_dot = Ds_dot * DmI;
            Matrix3d Ft = F.transpose();
            Matrix3d E_dot = 0.5 * (F_dot.transpose() * F + Ft * F_dot);

            double trE = E.trace();
            double trEdot = E_dot.trace();
            Matrix3d S = (m_lambda * trE + m_phi * trEdot) * I
                       + (2.0 * m_mu) * E + (2.0 * m_psi) * E_dot;

            Matrix3d H = -(F * S) * m_volDmInvT[t];

            Vector3d h0 = H.col(0), h1 = H.col(1), h2 = H.col(2);
            m_threadForces[tid][tet[0]] -= h0 + h1 + h2;
            m_threadForces[tid][tet[1]] += h0;
            m_threadForces[tid][tet[2]] += h1;
            m_threadForces[tid][tet[3]] += h2;
        }

        // Reduce
        #pragma omp for schedule(static)
        for (int i = 0; i < numVerts; ++i) {
            for (int tid2 = 0; tid2 < numThreads; ++tid2) {
                forces[i] += m_threadForces[tid2][i];
            }
        }
    }
}

void Simulation::resolveCollisions()
{
    const int numVerts = m_vertices.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numVerts; ++i) {
        // Ground plane collision (y = m_groundY)
        if (m_vertices[i].y() < m_groundY) {
            m_vertices[i].y() = m_groundY + 1e-4;

            Vector3d normal(0, 1, 0);
            double vn = m_velocities[i].dot(normal);
            Vector3d v_normal = vn * normal;
            Vector3d v_tangent = m_velocities[i] - v_normal;

            if (vn < 0) {
                m_velocities[i] = -m_restitution * v_normal + m_friction * v_tangent;
            }
        }

        // Sphere collision
        if (m_sphereEnabled) {
            Vector3d diff = m_vertices[i] - m_sphereCenter;
            double dist = diff.norm();
            if (dist < m_sphereRadius) {
                Vector3d normal = diff.normalized();
                m_vertices[i] = m_sphereCenter + m_sphereRadius * normal;

                double vn = m_velocities[i].dot(normal);
                Vector3d v_normal = vn * normal;
                Vector3d v_tangent = m_velocities[i] - v_normal;

                if (vn < 0) {
                    m_velocities[i] = -m_restitution * v_normal + m_friction * v_tangent;
                }
            }
        }
    }
}

void Simulation::loadMesh(const std::string &meshPath)
{
    std::vector<Vector3d> vertices;
    std::vector<Vector4i> tets;

    if (MeshLoader::loadTetMesh(meshPath, vertices, tets))
    {
        std::vector<Vector3i> faces;

        std::map<std::vector<int>, int> faceCount;
        std::map<std::vector<int>, Vector3i> faceOrientation;

        for (const auto &tet : tets)
        {
            int localFaces[4][3] = {{tet[1], tet[0], tet[2]},
                                    {tet[2], tet[0], tet[3]},
                                    {tet[3], tet[1], tet[2]},
                                    {tet[3], tet[0], tet[1]}};

            for (int i = 0; i < 4; ++i)
            {
                std::vector<int> f = {localFaces[i][0], localFaces[i][1], localFaces[i][2]};
                Vector3i originalFace(f[0], f[1], f[2]);

                std::sort(f.begin(), f.end());

                faceCount[f]++;
                faceOrientation[f] = originalFace;
            }
        }

        for (auto const &[key, count] : faceCount)
        {
            if (count == 1)
            {
                faces.push_back(faceOrientation[key]);
            }
        }

        m_shape.init(vertices, faces, tets);

        m_vertices = vertices;
        m_restVertices = vertices;
        m_tets = tets;
    }
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
    m_vertices = m_restVertices;
    std::fill(m_velocities.begin(), m_velocities.end(), Vector3d::Zero());
    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_vertices);
}


void Simulation::update(double seconds)
{
    if (m_vertices.empty() || m_paused) return;

    double remaining = seconds;
    while (remaining > 0.0) {
        double dt = std::min(m_dt, remaining);
        remaining -= dt;

        const int n = m_vertices.size();
        const double hdt = 0.5 * dt;
        const double dt6 = dt / 6.0;

        // --- RK4 with pre-allocated buffers ---
        // k1
        computeForces(m_vertices, m_velocities, m_forces);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            m_k1x[i] = m_velocities[i];
            m_k1v[i] = m_forces[i] * m_invMasses[i];
            m_tmpPos[i] = m_vertices[i]   + hdt * m_k1x[i];
            m_tmpVel[i] = m_velocities[i] + hdt * m_k1v[i];
        }

        // k2
        computeForces(m_tmpPos, m_tmpVel, m_forces);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            m_k2x[i] = m_tmpVel[i];
            m_k2v[i] = m_forces[i] * m_invMasses[i];
            m_tmpPos[i] = m_vertices[i]   + hdt * m_k2x[i];
            m_tmpVel[i] = m_velocities[i] + hdt * m_k2v[i];
        }

        // k3
        computeForces(m_tmpPos, m_tmpVel, m_forces);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            m_k3x[i] = m_tmpVel[i];
            m_k3v[i] = m_forces[i] * m_invMasses[i];
            m_tmpPos[i] = m_vertices[i]   + dt * m_k3x[i];
            m_tmpVel[i] = m_velocities[i] + dt * m_k3v[i];
        }

        // k4
        computeForces(m_tmpPos, m_tmpVel, m_forces);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            m_k4x[i] = m_tmpVel[i];
            m_k4v[i] = m_forces[i] * m_invMasses[i];
            m_vertices[i]   += dt6 * (m_k1x[i] + 2.0*m_k2x[i] + 2.0*m_k3x[i] + m_k4x[i]);
            m_velocities[i] += dt6 * (m_k1v[i] + 2.0*m_k2v[i] + 2.0*m_k3v[i] + m_k4v[i]);
        }

        resolveCollisions();
    }

    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_vertices);
}

void Simulation::draw(Shader *shader)
{
    m_shape.draw(shader);
    m_ground.draw(shader);
    if (m_sphereEnabled) {
        m_sphere.draw(shader);
    }
}

void Simulation::toggleWire()
{
    m_shape.toggleWireframe();
}

void Simulation::initGround()
{
    std::vector<Vector3d> groundVerts;
    std::vector<Vector3i> groundFaces;
    groundVerts.emplace_back(-50, 0, -50);
    groundVerts.emplace_back(-50, 0, 50);
    groundVerts.emplace_back(50, 0, 50);
    groundVerts.emplace_back(50, 0, -50);
    groundFaces.emplace_back(0, 1, 2);
    groundFaces.emplace_back(0, 2, 3);
    m_ground.init(groundVerts, groundFaces);
}

void Simulation::initSphere()
{
    std::vector<Vector3d> vertices;
    std::vector<Vector4i> tets;

    if (!MeshLoader::loadTetMesh(m_sphereMeshPath, vertices, tets)) {
        std::cerr << "Failed to load sphere obstacle mesh: " << m_sphereMeshPath << std::endl;
        return;
    }

    // Scale and position vertices to match the collision sphere
    for (auto &v : vertices) {
        v = m_sphereCenter + v * m_sphereRadius;
    }

    // Extract surface faces
    std::map<std::vector<int>, int> faceCount;
    std::map<std::vector<int>, Vector3i> faceOrientation;

    for (const auto &tet : tets) {
        int localFaces[4][3] = {{tet[1], tet[0], tet[2]},
                                {tet[2], tet[0], tet[3]},
                                {tet[3], tet[1], tet[2]},
                                {tet[3], tet[0], tet[1]}};
        for (int i = 0; i < 4; ++i) {
            std::vector<int> f = {localFaces[i][0], localFaces[i][1], localFaces[i][2]};
            Vector3i originalFace(f[0], f[1], f[2]);
            std::sort(f.begin(), f.end());
            faceCount[f]++;
            faceOrientation[f] = originalFace;
        }
    }

    std::vector<Vector3i> faces;
    for (auto const &[key, count] : faceCount) {
        if (count == 1) {
            faces.push_back(faceOrientation[key]);
        }
    }

    m_sphere.init(vertices, faces);
}
