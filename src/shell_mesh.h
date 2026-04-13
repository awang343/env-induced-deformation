#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

// Triangle mesh + half-edge-lite topology for thin shell simulation.
// All topology is built once by buildTopology() after vertices/faces are set.
struct ShellMesh
{
    // One (face, local-opposite-vertex) record per edge side.
    struct EdgeFaceRef
    {
        int face = -1;
        int localOppVtx = -1;
    };

    // ---- Geometry ----
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;

    // ---- Topology (built once) ----
    std::vector<Eigen::Vector2i> edges;                    // (v0,v1) per edge, v0<v1
    std::vector<Eigen::Vector3i> faceEdges;                // per face: edge ids, [i] opposite local vertex i
    std::vector<Eigen::Vector3i> faceNeighbors;            // per face: neighbor face ids (-1 if boundary)
    std::vector<std::array<EdgeFaceRef, 2>> edgeFaces;     // per edge: up to 2 incident (face, oppVtx)
    std::vector<int> vertexFaceOffsets;                     // CSR: vertex v -> faces in vertexFaceList[offsets[v]..offsets[v+1])
    std::vector<int> vertexFaceList;

    // ---- Methods ----
    // Load triangle mesh from OBJ file (v + f lines, 1-indexed).
    bool load(const std::string &objPath);

    // Build edges, face-edges, face-neighbors, edge-faces, and vertex-face CSR.
    // Must be called after vertices/faces are populated.
    void buildTopology();

    int numVerts() const { return static_cast<int>(vertices.size()); }
    int numFaces() const { return static_cast<int>(faces.size()); }
    int numEdges() const { return static_cast<int>(edges.size()); }
};
