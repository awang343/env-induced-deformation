#include "halfedge.h"
#include <iostream>
#include <unordered_set>

bool HalfEdgeRepr::edgeFlipValid(HalfEdge *he)
{
    Vertex *v_t = he->next->next->vertex;
    Vertex *v_b = he->twin->next->next->vertex;

    HalfEdge *start = v_t->halfEdge;
    HalfEdge *current = start;

    do
    {
        if (current->twin->vertex == v_b)
        {
            return false;
        }
        current = current->twin->next;
    } while (current != start);

    return degree(he->vertex) != 3 && degree(he->twin->vertex) != 3;
}

bool HalfEdgeRepr::edgeCollapseValid(HalfEdge *he)
{
    if (!he->twin) return false;
    Vertex *u = he->vertex;
    Vertex *v = he->twin->vertex;
    Vertex *v_t = he->next->next->vertex;
    Vertex *v_b = he->twin->next->next->vertex;

    if (degree(v_t) <= 3 || degree(v_b) <= 3)
    {
        return false;
    }

    // Make sure v_t and v_b are the only shared neighbors of u and v
    std::unordered_set<int> neighbors_u;

    HalfEdge *start_u = u->halfEdge;
    HalfEdge *current_u = start_u;
    do
    {
        Vertex *nb = current_u->next->vertex;
        if (nb->id != v->id)
            neighbors_u.insert(nb->id);
        if (!current_u->twin) return false;
        current_u = current_u->twin->next;
    } while (current_u != start_u);

    HalfEdge *start_v = v->halfEdge;
    HalfEdge *current_v = start_v;

    do
    {
        Vertex *nb = current_v->next->vertex;
        if (nb->id == u->id)
        {
            if (!current_v->twin) return false;
            current_v = current_v->twin->next;
            continue;
        }
        if (neighbors_u.count(nb->id) > 0 && nb->id != v_t->id && nb->id != v_b->id)
        {
            return false;
        }
        if (!current_v->twin) return false;
        current_v = current_v->twin->next;
    } while (current_v != start_v);

    // Check for inverted triangles after collapse
    Eigen::Vector3f collapsePos = (u->position + v->position) / 2; // simulate collapsing u->v

    auto createsInvertedTriangle = [&](Eigen::Vector3f to_collapse, Eigen::Vector3f p1,
                                       Eigen::Vector3f p2) -> bool
    {
        Eigen::Vector3f n_before = ((p1 - to_collapse).cross(p2 - to_collapse)).normalized();
        Eigen::Vector3f n_after = ((p1 - collapsePos).cross(p2 - collapsePos)).normalized();
        return n_before.dot(n_after) < 0.0f; // flipped normal => inverted
    };

    // 1. Check all faces incident to u
    HalfEdge *startHeU = u->halfEdge;
    HalfEdge *heU = startHeU;
    do
    {
        Face *f = heU->face;
        if (f && !(heU->next->vertex == v || heU->next->next->vertex == v))
        {
            Eigen::Vector3f p1 = heU->next->vertex->position;
            Eigen::Vector3f p2 = heU->next->next->vertex->position;
            if (createsInvertedTriangle(u->position, p1, p2))
                return false;
        }
        if (!heU->twin) return false;
        heU = heU->twin->next;
    } while (heU != startHeU);

    // 2. Check all faces incident to v that do not contain u
    HalfEdge *startHeV = v->halfEdge;
    HalfEdge *heV = startHeV;
    do
    {
        Face *f = heV->face;
        if (f && !(heV->next->vertex == u || heV->next->next->vertex == u))
        {
            Eigen::Vector3f p1 = heV->next->vertex->position;
            Eigen::Vector3f p2 = heV->next->next->vertex->position;
            if (createsInvertedTriangle(v->position, p1, p2))
                return false;
        }
        if (!heV->twin) return false;
        heV = heV->twin->next;
    } while (heV != startHeV);

    return true;
}

void HalfEdgeRepr::validate()
{
    std::cout << "Sweeping before validation..." << std::endl;
    sweep();

    std::cout << "Validating half-edge data structure..." << std::endl;
    std::cout << "\tNumber of vertices: " << _vertices.size() << std::endl;
    std::cout << "\tNumber of faces: " << _faces.size() << std::endl;
    std::cout << "\tNumber of half-edges: " << _half_edges.size() << std::endl;
    std::cout << "\tNumber of edges: " << _edges.size() << std::endl;

    assert(_half_edges.size() % 6 == 0);
    assert(_half_edges.size() / 3 == _faces.size());
    assert(_edges.size() == _half_edges.size() / 2);

    // Check that each undirected edge has exactly 2 half-edges with opposite orientations
    // key -> {count, orientationFlag}
    // orientationFlag:
    //   0 = first half-edge seen is (a -> b)
    //   1 = first half-edge seen is (b -> a)
    std::unordered_map<uint64_t, std::pair<int, int>> edgeMap;
    for (const auto &he : _half_edges)
    {
        // Basic consistency checks
        assert(he->vertex != nullptr);
        assert(_vertices[he->vertex->id].get() == he->vertex);

        assert(he->face != nullptr);
        assert(_faces[he->face->id].get() == he->face);

        assert(he->edge != nullptr);
        assert(_edges[he->edge->id].get() == he->edge);

        assert(he->next != nullptr);
        assert(he->twin != nullptr);

        assert(he->vertex != he->next->vertex);

        if (he->next->next->next != he.get())
        {
            std::cerr << "Error: Half-edge does not form a triangle." << std::endl;
            std::cerr << "Half-edge: " << he.get() << std::endl;
            std::cerr << "Next Half-edge: " << he->next << std::endl;
            std::cerr << "Next Next Half-edge: " << he->next->next << std::endl;
            std::cerr << "Next Next Next Half-edge: " << he->next->next->next << std::endl;
        }
        assert(he->next->next->next == he.get());
        assert(he->twin->twin == he.get());
        assert(he->edge == he->twin->edge);

        int v0 = he->vertex->id;
        int v1 = he->next->vertex->id;

        auto [a, b] = std::minmax(v0, v1);
        uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);

        bool isForward = (v0 == a && v1 == b);

        auto &entry = edgeMap[key];

        if (entry.first == 0)
        {
            // First time we see this undirected edge
            entry.first = 1;
            entry.second = isForward ? 0 : 1;
        }
        else
        {
            // Second time — must be opposite direction
            entry.first++;

            bool firstWasForward = (entry.second == 0);

            if (entry.first > 2)
            {
                std::cerr << "Error: More than 2 half-edges found for edge (" << a << ", " << b
                          << ")" << std::endl;
                std::cerr << "Position of vertex " << a << ": "
                          << _vertices[a]->position.transpose() << std::endl;
                std::cerr << "Position of vertex " << b << ": "
                          << _vertices[b]->position.transpose() << std::endl;
            }
            assert(entry.first <= 2);             // at most 2 half-edges
            assert(firstWasForward != isForward); // must be opposite orientation
        }

        assert(!he->deleted);
    }

    // Check that all edges have exactly 2 half-edges which are twins
    for (const auto &edge : _edges)
    {
        assert(_edges[edge->id].get() == edge.get());

        assert(edge->halfEdge != nullptr && !edge->halfEdge->deleted);
        assert(edge->halfEdge->twin != nullptr && !edge->halfEdge->twin->deleted);
        assert(edge->halfEdge->edge == edge->halfEdge->twin->edge);

        assert(!edge->deleted);
    }

    // Check vertex half-edge pointers
    for (const auto &vertex : _vertices)
    {
        assert(_vertices[vertex->id].get() == vertex.get());

        assert(vertex->halfEdge != nullptr && !vertex->halfEdge->deleted);
        assert(vertex->halfEdge->vertex == vertex.get());

        assert(!vertex->deleted);
    }

    // Check face assignments
    for (const auto &face : _faces)
    {
        assert(_faces[face->id].get() == face.get());

        assert(face->halfEdge != nullptr && !face->halfEdge->deleted);
        HalfEdge *he = face->halfEdge;
        do
        {
            assert(he->face == face.get());
            he = he->next;
        } while (he != face->halfEdge);

        assert(!face->deleted);
    }

    std::cout << "Validation passed!" << std::endl << std::endl;
}
