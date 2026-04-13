#pragma once

#include <vector>
#include "Eigen/Dense"
#include "Eigen/StdVector"

class MeshLoader
{
public:
    // Loads a triangle mesh from a simple text file. Each line is one of:
    //   v  x y z          vertex position (three floats)
    //   f  i0 i1 i2       triangle (three 0-indexed vertex ids)
    // Any other line is ignored.
    static bool loadTriMesh(const std::string &filepath,
                            std::vector<Eigen::Vector3d> &vertices,
                            std::vector<Eigen::Vector3i> &faces);
private:
    MeshLoader();
};
