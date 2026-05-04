#include "embedding.h"

#include <cmath>
#include <limits>

using namespace Eigen;

// Compute closest point on triangle (p0,p1,p2) to query point q.
// Returns barycentric coordinates and squared distance.
static double closestPointOnTriangle(
    const Vector3d &p0, const Vector3d &p1, const Vector3d &p2,
    const Vector3d &q, Vector3d &bary)
{
    Vector3d e1 = p1 - p0, e2 = p2 - p0, ep = q - p0;
    double d11 = e1.dot(e1), d12 = e1.dot(e2), d22 = e2.dot(e2);
    double dp1 = ep.dot(e1), dp2 = ep.dot(e2);
    double denom = d11 * d22 - d12 * d12;

    double v, w;
    if (std::abs(denom) < 1e-16) {
        // Degenerate triangle — fall back to nearest vertex.
        double d0 = (q - p0).squaredNorm();
        double d1 = (q - p1).squaredNorm();
        double d2 = (q - p2).squaredNorm();
        if (d0 <= d1 && d0 <= d2) { bary = {1,0,0}; return d0; }
        if (d1 <= d2)             { bary = {0,1,0}; return d1; }
        bary = {0,0,1}; return d2;
    }

    v = (d22 * dp1 - d12 * dp2) / denom;
    w = (d11 * dp2 - d12 * dp1) / denom;
    double u = 1.0 - v - w;

    // Clamp to triangle if outside.
    if (u < 0) { u = 0; double s = d22 > 1e-16 ? dp2/d22 : 0; w = std::clamp(s, 0.0, 1.0); v = 1-w; }
    else if (v < 0) { v = 0; double t = (e2-e1).squaredNorm(); double s = t > 1e-16 ? (q-p1).dot(p2-p1)/t : 0; w = std::clamp(s, 0.0, 1.0); u = 1-w; }
    else if (w < 0) { w = 0; double s = d11 > 1e-16 ? dp1/d11 : 0; v = std::clamp(s, 0.0, 1.0); u = 1-v; }

    bary = {u, v, w};
    Vector3d closest = u * p0 + v * p1 + w * p2;
    return (q - closest).squaredNorm();
}

void computeEmbedding(const ShellMesh &displayMesh,
                      const ShellMesh &physicsMesh,
                      std::vector<BarycentricEmbed> &embed)
{
    const int nD = displayMesh.numVerts();
    const int nF = physicsMesh.numFaces();
    embed.resize(nD);

    for (int i = 0; i < nD; ++i) {
        const Vector3d &q = displayMesh.vertices[i];
        double bestDist = std::numeric_limits<double>::max();
        int bestFace = 0;
        Vector3d bestBary = {1.0/3, 1.0/3, 1.0/3};

        for (int f = 0; f < nF; ++f) {
            const auto &tri = physicsMesh.faces[f];
            Vector3d bary;
            double dist = closestPointOnTriangle(
                physicsMesh.vertices[tri[0]],
                physicsMesh.vertices[tri[1]],
                physicsMesh.vertices[tri[2]],
                q, bary);
            if (dist < bestDist) {
                bestDist = dist;
                bestFace = f;
                bestBary = bary;
            }
        }

        embed[i].face = bestFace;
        embed[i].bary = bestBary;
    }
}
