#pragma once

#include <Eigen/Dense>
#include <vector>

// Simplify a triangle mesh using QEM edge collapse.
// ratio: fraction of faces to remove (0 = none, 1 = all).
void simplifyMesh(const std::vector<Eigen::Vector3d> &inVerts,
                  const std::vector<Eigen::Vector3i> &inFaces,
                  double ratio,
                  std::vector<Eigen::Vector3d> &outVerts,
                  std::vector<Eigen::Vector3i> &outFaces);
