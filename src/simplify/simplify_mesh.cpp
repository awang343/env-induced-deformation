#include "simplify_mesh.h"
#include "mesh.h"

#include <algorithm>
#include <iostream>

using namespace Eigen;

void simplifyMesh(const std::vector<Vector3d> &inVerts,
                  const std::vector<Vector3i> &inFaces,
                  double ratio,
                  std::vector<Vector3d> &outVerts,
                  std::vector<Vector3i> &outFaces)
{
    // Convert Vector3d → Vector3f for the simplification library.
    std::vector<Vector3f> vf(inVerts.size());
    for (size_t i = 0; i < inVerts.size(); ++i)
        vf[i] = inVerts[i].cast<float>();

    Mesh m;
    m.initFromVectors(vf, inFaces);

    int numToRemove = std::clamp((int)(ratio * inFaces.size()), 0, (int)inFaces.size() - 4);
    if (numToRemove > 0) {
        std::cout << "Simplifying: " << inFaces.size() << " → "
                  << (inFaces.size() - numToRemove) << " faces" << std::endl;
        try {
            m.simplify(numToRemove);
        } catch (const std::exception &e) {
            std::cerr << "Simplification failed: " << e.what()
                      << " — using original mesh" << std::endl;
            outVerts = inVerts;
            outFaces = inFaces;
            return;
        }
    }

    // Convert back Vector3f → Vector3d.
    const auto &rv = m.vertices();
    const auto &rf = m.faces();
    outVerts.resize(rv.size());
    for (size_t i = 0; i < rv.size(); ++i)
        outVerts[i] = rv[i].cast<double>();
    outFaces = rf;
}
