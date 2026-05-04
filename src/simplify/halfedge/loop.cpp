#include "halfedge.h"

float compute_u(int deg)
{
    if (deg == 3)
    {
        return 3.0f / 16.0f;
    }
    else
    {
        return (0.625f - powf(0.375f + 0.25f * cosf(2 * M_PI / deg), 2)) / deg;
    }
}

void HalfEdgeRepr::loop_subdivision()
{
    // Implement loop subdivision
    int num_old_vertices = _vertices.size();
    int num_old_edges = _edges.size();

    // Compute new vertex positions
    std::vector<Eigen::Vector3f> updated_positions(_vertices.size());
    for (size_t i = 0; i < _vertices.size(); ++i)
    {
        Eigen::Vector3f sum_neighbors(0.0f, 0.0f, 0.0f);
        int deg = 0;

        HalfEdge *start = _vertices[i]->halfEdge;
        HalfEdge *he = start;
        do
        {
            sum_neighbors += he->twin->vertex->position;
            he = he->twin->next; // move around the vertex
            deg++;
        } while (he != start);

        float u = compute_u(deg);
        updated_positions[i] = (1 - deg * u) * _vertices[i]->position + u * sum_neighbors;
    }

    std::vector<Eigen::Vector3f> new_positions(num_old_edges);
    for (size_t i = 0; i < num_old_edges; ++i)
    {
        HalfEdge *he = _edges[i]->halfEdge;
        Eigen::Vector3f new_pos =
            0.375f * (he->vertex->position + he->twin->vertex->position) +
            0.125f * (he->next->next->vertex->position + he->twin->next->next->vertex->position);
        new_positions[i] = new_pos;
    }

    // Create new vertices and assign positions to all vertices
    for (size_t i = 0; i < _vertices.size(); ++i)
    {
        _vertices[i]->position = updated_positions[i];
    }

    for (int i = 0; i < num_old_edges; ++i)
    {
        edgeSplit(_edges[i]->halfEdge);
        _edges[i]->halfEdge->vertex->position = new_positions[i];
    }

    // Flip edges to get the 4-triangle split per face
    for (int i = num_old_edges; i < _edges.size(); ++i)
    {
        // Check if the edge connects old to new vertex
        bool connects_old_to_new = (_edges[i]->halfEdge->vertex->id < num_old_vertices) ^
                                   (_edges[i]->halfEdge->twin->vertex->id < num_old_vertices);

        if (_edges[i]->is_new && connects_old_to_new)
        {
            edgeFlip(_edges[i]->halfEdge);
        }
    }
}
