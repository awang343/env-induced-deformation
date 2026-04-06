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

    void loadMesh(const std::string &meshPath);

    // Mesh data
    std::vector<Eigen::Vector3d> m_vertices;
    std::vector<Eigen::Vector3d> m_restVertices;
    std::vector<Eigen::Vector4i> m_tets;

    bool m_paused = true;
    bool m_parallel = true;
    std::string m_configPath;
};
