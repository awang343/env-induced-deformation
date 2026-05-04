#include "halfedge.h"
#include <iostream>

void update_edge_lengths(std::deque<std::unique_ptr<Edge>> &edges)
{
    for (const auto &edgePtr : edges)
    {
        Eigen::Vector3f v1 = edgePtr->halfEdge->vertex->position;
        Eigen::Vector3f v2 = edgePtr->halfEdge->twin->vertex->position;

        edgePtr->length = (v1 - v2).norm();
    }
}

float average_edge_length(std::deque<std::unique_ptr<Edge>> &edges)
{
    // Compute edge lengths and return average edge length
    float totalLength = 0.0f;

    for (const auto &edgePtr : edges)
    {
        totalLength += edgePtr->length;
    }

    return totalLength / edges.size();
}

void compute_face_normals(std::deque<std::unique_ptr<Face>> &faces)
{
    for (const auto &facePtr : faces)
    {
        HalfEdge *he = facePtr->halfEdge;
        Eigen::Vector3f v0 = he->vertex->position;
        Eigen::Vector3f v1 = he->next->vertex->position;
        Eigen::Vector3f v2 = he->next->next->vertex->position;

        facePtr->normal = (v1 - v0).cross(v2 - v0);
    }
}

void compute_vertex_normals(std::deque<std::unique_ptr<Vertex>> &vertices)
{
    for (const auto &vertexPtr : vertices)
    {
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        HalfEdge *startHe = vertexPtr->halfEdge;
        HalfEdge *currentHe = startHe;

        do
        {
            normal += currentHe->face->normal;
            currentHe = currentHe->twin->next;
        } while (currentHe != startHe);

        vertexPtr->normal = normal.normalized();
    }
}

void HalfEdgeRepr::isotropic_remesh(int numIterations, double tangentialSmoothingWeight)
{
    std::cout << numIterations
              << " iterations of isotropic remeshing with tangential smoothing weight "
              << tangentialSmoothingWeight << std::endl;

    update_edge_lengths(_edges);
    float avgEdgeLength = average_edge_length(_edges);

    for (int iter = 0; iter < numIterations; ++iter)
    {
        // Step 1: Edge split
        update_edge_lengths(_edges);
        for (size_t i = 0; i < _edges.size(); ++i)
        {
            if (!_edges[i]->deleted && _edges[i]->length > 4.0f / 3.0f * avgEdgeLength)
            {
                edgeSplit(_edges[i]->halfEdge);
            }
        }

        // Step 2: Edge collapse
        update_edge_lengths(_edges);
        for (size_t i = 0; i < _edges.size(); ++i)
        {
            if (!_edges[i]->halfEdge->deleted && _edges[i]->length < 3.0f / 5.0f * avgEdgeLength &&
                edgeCollapseValid(_edges[i]->halfEdge))
            {
                edgeCollapse(_edges[i]->halfEdge);
            }
        }
        sweep();

        // Step 3: Edge flipping
        for (int i = 0; i < _edges.size(); ++i)
        {
            // Flip edge if it improves the valence of the vertices from 6
            if (!edgeFlipValid(_edges[i]->halfEdge))
                continue;

            int vr_deg_impr = degree(_edges[i]->halfEdge->vertex) > 6 ? 1 : -1;
            int vl_deg_impr = degree(_edges[i]->halfEdge->twin->vertex) > 6 ? 1 : -1;
            int vt_deg_impr = degree(_edges[i]->halfEdge->next->next->vertex) < 6 ? 1 : -1;
            int vb_deg_impr = degree(_edges[i]->halfEdge->twin->next->next->vertex) < 6 ? 1 : -1;

            if (vr_deg_impr + vl_deg_impr + vt_deg_impr + vb_deg_impr > 0)
            {
                edgeFlip(_edges[i]->halfEdge);
            }
        }

        // // Step 4: Tangential smoothing
        for (int c = 0; c < 5; ++c)
        {
            std::vector<Eigen::Vector3f> newPositions(_vertices.size());

            compute_face_normals(_faces);
            compute_vertex_normals(_vertices);

            for (int i = 0; i < _vertices.size(); ++i)
            {
                Eigen::Vector3f p = _vertices[i]->position;
                Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
                int n = 0;

                // Loop over neighboring vertices
                HalfEdge *heStart = _vertices[i]->halfEdge;
                HalfEdge *he = heStart;
                do
                {
                    centroid += he->twin->vertex->position;
                    n++;
                    he = he->twin->next;
                } while (he != heStart);

                if (n == 0)
                {
                    newPositions[i] = p; // No neighbors, keep original position
                    continue;
                }

                centroid /= float(n);

                // Compute tangential displacement
                Eigen::Vector3f displacement = centroid - p;
                Eigen::Vector3f normal = _vertices[i]->normal;
                Eigen::Vector3f tangentialDisp = displacement - displacement.dot(normal) * normal;

                // Apply weighted tangential smoothing
                newPositions[i] = p + tangentialSmoothingWeight * tangentialDisp;
            }

            for (int i = 0; i < _vertices.size(); ++i)
            {
                _vertices[i]->position = newPositions[i];
            }
        }

        update_edge_lengths(_edges);
    }
}
