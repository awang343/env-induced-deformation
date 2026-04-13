#include "shell_mesh.h"
#include "graphics/meshloader.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <utility>

using namespace Eigen;

bool ShellMesh::load(const std::string &objPath)
{
    std::vector<Vector3d> verts;
    std::vector<Vector3i> tris;
    if (!MeshLoader::loadTriMesh(objPath, verts, tris)) {
        std::cerr << "Failed to load mesh: " << objPath << std::endl;
        return false;
    }
    vertices = std::move(verts);
    faces    = std::move(tris);
    return true;
}

void ShellMesh::buildTopology()
{
    const int nF = numFaces();
    const int nV = numVerts();

    std::map<std::pair<int,int>, int> edgeIndex;
    edges.clear();
    edgeFaces.clear();
    faceEdges.assign(nF, Vector3i(-1, -1, -1));

    auto edgeKey = [](int a, int b) {
        return std::make_pair(std::min(a, b), std::max(a, b));
    };

    for (int f = 0; f < nF; ++f) {
        const Vector3i &tri = faces[f];
        for (int i = 0; i < 3; ++i) {
            int a = tri[(i + 1) % 3];
            int b = tri[(i + 2) % 3];
            auto key = edgeKey(a, b);
            auto it = edgeIndex.find(key);
            int eid;
            if (it == edgeIndex.end()) {
                eid = static_cast<int>(edges.size());
                edgeIndex[key] = eid;
                edges.emplace_back(key.first, key.second);
                edgeFaces.emplace_back();
                edgeFaces.back()[0] = EdgeFaceRef{f, i};
            } else {
                eid = it->second;
                auto &refs = edgeFaces[eid];
                assert(refs[1].face == -1 && "non-manifold edge in input mesh");
                refs[1] = EdgeFaceRef{f, i};
            }
            faceEdges[f][i] = eid;
        }
    }

    faceNeighbors.assign(nF, Vector3i(-1, -1, -1));
    for (int f = 0; f < nF; ++f) {
        for (int i = 0; i < 3; ++i) {
            int eid = faceEdges[f][i];
            const auto &refs = edgeFaces[eid];
            faceNeighbors[f][i] = (refs[0].face == f) ? refs[1].face : refs[0].face;
        }
    }

    vertexFaceOffsets.assign(nV + 1, 0);
    for (int f = 0; f < nF; ++f)
        for (int i = 0; i < 3; ++i) vertexFaceOffsets[faces[f][i] + 1]++;
    for (int v = 0; v < nV; ++v)
        vertexFaceOffsets[v + 1] += vertexFaceOffsets[v];

    vertexFaceList.assign(vertexFaceOffsets.back(), 0);
    std::vector<int> cursor(nV, 0);
    for (int f = 0; f < nF; ++f)
        for (int i = 0; i < 3; ++i) {
            int v = faces[f][i];
            vertexFaceList[vertexFaceOffsets[v] + cursor[v]++] = f;
        }
}
