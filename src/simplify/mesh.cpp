#include "mesh.h"

using namespace Eigen;
using namespace std;

void Mesh::initFromVectors(const vector<Vector3f> &vertices, const vector<Vector3i> &faces)
{
    _vertices = vertices;
    _faces = faces;
}

void Mesh::toHalfEdge()
{
    _halfEdgeRepr.initFromVectors(_vertices, _faces);
}

void Mesh::fromHalfEdge()
{
    _halfEdgeRepr.outputToVectors(_vertices, _faces);
}

void Mesh::simplify(int numFacesToRemove)
{
    toHalfEdge();
    _halfEdgeRepr.quadric_simplification(numFacesToRemove);
    fromHalfEdge();
}
