#include "simulation.h"
#include "graphics/meshloader.h"

#include <QSettings>

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

using namespace Eigen;

Simulation::Simulation()
{
    omp_set_num_threads(std::max(1, omp_get_num_procs() - 2));
}

void Simulation::init(const std::string &config_path)
{
    m_configPath = config_path;
    QSettings cfg(QString::fromStdString(m_configPath), QSettings::IniFormat);

    cfg.beginGroup("simulation");
    std::string meshPath = cfg.value("mesh").toString().toStdString();
    cfg.endGroup();

    loadMesh(meshPath);

    m_restVertices = m_vertices;

    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_vertices);
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
    m_shape.setModelMatrix(Affine3f::Identity());
    m_shape.setVertices(m_vertices);
}

void Simulation::update(double seconds)
{
    if (m_vertices.empty() || m_paused) return;

    // TODO: implement paper
}

void Simulation::draw(Shader *shader)
{
    m_shape.draw(shader);
}

void Simulation::toggleWire()
{
    m_shape.toggleWireframe();
}
