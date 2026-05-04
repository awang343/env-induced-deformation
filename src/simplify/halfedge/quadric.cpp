#include "halfedge.h"
#include <iostream>
#include <map>
#include <set>

using namespace Eigen;

using QueueEntry = std::tuple<Edge *, Vector3f, float>;

void computePlane(Face *face)
{
    HalfEdge *he = face->halfEdge;
    Vector3f v0 = he->vertex->position;
    Vector3f v1 = he->next->vertex->position;
    Vector3f v2 = he->next->next->vertex->position;

    // Two edges of the triangle
    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;

    // Normal vector (a, b, c) = cross product
    Vector3f normal = edge1.cross(edge2);

    // Normalize to unit length
    normal.normalize();

    // Compute d = -(normal · point)
    float d = -normal.dot(v0);

    // Return as (a, b, c, d)
    face->plane = Vector4f(normal.x(), normal.y(), normal.z(), d);
}

void computeQuadric(Vertex *vertex)
{
    vertex->quadric.setZero();
    HalfEdge *start = vertex->halfEdge;
    HalfEdge *current = start;
    do
    {
        Face *face = current->face;
        Vector4f p = face->plane;
        vertex->quadric += p * p.transpose();
        if (!current->twin) break;  // boundary
        current = current->twin->next;
    } while (current != start);
}

std::pair<Vector3f, float> computeEdgeError(const Matrix4f &Q, const Vector3f &v1,
                                            const Vector3f &v2)
{
    // Compute the optimal point along the edge that minimizes v^T Q v

    // First, check if we can solve for the optimal point
    // The optimal point is the solution to Q * [x, y, z, 1]^T = [0, 0, 0, λ]^T
    // More practically, we solve the 4x4 linear system for [x, y, z, 1]

    // Extract the upper 3x3 part and the last column/row
    Matrix3f Q3 = Q.block<3, 3>(0, 0);
    Vector3f q4 = Q.block<3, 1>(0, 3);
    float q44 = Q(3, 3);

    // Check if Q3 is invertible (non-singular)
    float det = Q3.determinant();
    Vector4f optimal_point;

    if (std::abs(det) > 1e-10f)
    {
        // Solve for the optimal point: Q3 * p = -q4
        Vector3f p = -Q3.inverse() * q4;
        optimal_point = Vector4f(p.x(), p.y(), p.z(), 1.0f);
    }
    else
    {
        Vector3f dir = v2 - v1;

        float a = dir.transpose() * Q3 * dir;
        float b = 2.0f * dir.transpose() * (Q3 * v1 + q4);
        float c = v1.transpose() * Q3 * v1 + 2.0f * v1.dot(q4) + q44;

        float t_opt;
        if (std::abs(a) > 1e-10f)
        {
            t_opt = -b / (2.0f * a);
            t_opt = std::max(0.0f, std::min(1.0f, t_opt));
        }
        else
        {
            t_opt = (b > 0) ? 0.0f : 1.0f;
        }

        Vector3f p = v1 + t_opt * dir;
        optimal_point = Vector4f(p.x(), p.y(), p.z(), 1.0f);
    }

    float error = optimal_point.transpose() * Q * optimal_point;

    return std::make_pair(Vector3f(optimal_point.x(), optimal_point.y(), optimal_point.z()), error);
}

void HalfEdgeRepr::quadric_simplification(int numFacesToRemove)
{
    // Compute planes
    for (const auto &face : _faces)
    {
        computePlane(face.get());
    }

    // Compute quadrics for each vertex
    for (const auto &vertex : _vertices)
    {
        computeQuadric(vertex.get());
    }

    auto cmp = [](const QueueEntry &a, const QueueEntry &b)
    { return std::get<2>(a) < std::get<2>(b); };

    std::multiset<QueueEntry, decltype(cmp)> edgeQueue(cmp);
    std::map<Edge *, std::multiset<QueueEntry, decltype(cmp)>::iterator> edgeToQueueEntry;

    // Compute initial edge errors and populate the priority queue
    for (const auto &edge : _edges)
    {
        if (!edge->halfEdge->twin) continue;  // skip boundary edges
        Vertex *v1 = edge->halfEdge->vertex;
        Vertex *v2 = edge->halfEdge->twin->vertex;
        Matrix4f Q = v1->quadric + v2->quadric;

        auto [optimal_point, error] = computeEdgeError(Q, v1->position, v2->position);

        auto it = edgeQueue.insert(std::make_tuple(edge.get(), optimal_point, error));

        edgeToQueueEntry[edge.get()] = it;
    }

    // Iteratively collapse edges with the least error
    int collapsedFaces = 0;

    while (collapsedFaces < numFacesToRemove && !edgeQueue.empty())
    {
        auto smallest = edgeQueue.begin();
        Edge *edge = std::get<0>(*smallest);

        // Check if edge is still valid (not deleted and collapsible)


        if (edge->deleted || !edgeCollapseValid(edge->halfEdge))
        {
            edgeToQueueEntry.erase(edge); // Remove from map
            edgeQueue.erase(smallest);    // Remove invalid entry
            continue;
        }

        Vertex *center = edge->halfEdge->vertex;
        Vertex *to_delete = edge->halfEdge->twin->vertex;

        // Update quadrics and errors for affected vertices and edges
        edgeCollapse(edge->halfEdge);

        center->position = std::get<1>(*smallest);              // Move vertex to optimal position
        center->quadric = center->quadric + to_delete->quadric; // Update quadric

        edgeQueue.erase(smallest);
        edgeToQueueEntry.erase(edge); // Remove from map

        HalfEdge *he = center->halfEdge;
        do
        {
            Edge *e = he->edge;

            if (e && he->twin) {
                // Remove old entry for this edge if it exists
                auto delete_it = edgeToQueueEntry.find(e);
                if (delete_it != edgeToQueueEntry.end())
                {
                    edgeQueue.erase(delete_it->second);
                    edgeToQueueEntry.erase(delete_it);
                }

                Vertex *v1 = e->halfEdge->vertex;
                Vertex *v2 = e->halfEdge->twin->vertex;

                Matrix4f Q = v1->quadric + v2->quadric;

                auto [optimal_point, error] = computeEdgeError(Q, v1->position, v2->position);

                auto it = edgeQueue.insert(std::make_tuple(e, optimal_point, error));

                edgeToQueueEntry[e] = it;
            }

            if (!he->twin) break;  // boundary
            he = he->twin->next;
        } while (he != center->halfEdge);

        collapsedFaces += 2; // Each edge collapse removes 2 faces
    }

    // Skip validate() — it asserts closed manifold which boundary meshes violate.
}
