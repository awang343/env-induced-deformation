#pragma once

#include "shell_mesh.h"
#include <Eigen/Dense>
#include <vector>

struct BarycentricEmbed {
    int face;              // index into physics mesh faces
    Eigen::Vector3d bary;  // barycentric coordinates (u, v, w), u+v+w=1
};

// For each vertex in displayMesh, find the closest face in physicsMesh
// and compute barycentric coordinates. Uses rest configuration.
void computeEmbedding(const ShellMesh &displayMesh,
                      const ShellMesh &physicsMesh,
                      std::vector<BarycentricEmbed> &embed);
