#include "halfedge.h"
#include <iomanip>
#include <iostream>

int HalfEdgeRepr::degree(Vertex *v)
{
    int count = 0;
    HalfEdge *start = v->halfEdge;
    HalfEdge *current = start;
    do
    {
        count++;
        if (!current->twin) return count;  // boundary
        current = current->twin->next;
    } while (current != start);

    return count;
}

void HalfEdgeRepr::clear()
{
    _half_edges.clear();
    _vertices.clear();
    _faces.clear();
    _edges.clear();
}

void HalfEdgeRepr::sweep()
{
    // Cleanup function to remove deleted elements and renumber vertex IDs and reset edge flags

    _half_edges.erase(
        std::remove_if(_half_edges.begin(), _half_edges.end(), [](auto &h) { return h->deleted; }),
        _half_edges.end());
    _vertices.erase(
        std::remove_if(_vertices.begin(), _vertices.end(), [](auto &v) { return v->deleted; }),
        _vertices.end());
    _faces.erase(std::remove_if(_faces.begin(), _faces.end(), [](auto &f) { return f->deleted; }),
                 _faces.end());
    _edges.erase(std::remove_if(_edges.begin(), _edges.end(), [](auto &e) { return e->deleted; }),
                 _edges.end());

    // Renumber vertex IDs after deletion
    for (size_t i = 0; i < _half_edges.size(); ++i)
    {
        _half_edges[i]->id = i;
    }

    for (size_t i = 0; i < _vertices.size(); ++i)
    {
        _vertices[i]->id = i;
    }

    for (size_t i = 0; i < _faces.size(); ++i)
    {
        _faces[i]->id = i;
    }

    for (size_t i = 0; i < _edges.size(); ++i)
    {
        _edges[i]->id = i;
    }

    for (const auto &edge : _edges)
    {
        edge->is_new = false;
    }
}

void HalfEdgeRepr::dump()
{
    std::cout << "# vertices: id x y z\n";
    for (size_t i = 0; i < _vertices.size(); ++i)
    {
        const auto &v = _vertices[i];
        std::cout << v->id << " " << std::fixed << std::setprecision(6) << v->position.x() << " "
                  << v->position.y() << " " << v->position.z() << "\n";
    }

    std::cout << "# faces: vertex indices per face\n";
    for (const auto &f : _faces)
    {
        if (!f->halfEdge)
            continue;
        auto start = f->halfEdge;
        auto he = start;
        bool first = true;
        do
        {
            if (!first)
                std::cout << " ";
            std::cout << he->vertex->id;
            first = false;
            he = he->next;
        } while (he != start);
        std::cout << "\n";
    }

    std::cout << "# half-edges: from to twin_index deleted (-1 if none)\n";
    for (size_t i = 0; i < _half_edges.size(); ++i)
    {
        const auto &he = _half_edges[i];
        int from = he->vertex->id;
        int to = he->next ? he->next->vertex->id : -1;
        int twin_idx = -1;
        if (he->twin)
        {
            auto it = std::find_if(_half_edges.begin(), _half_edges.end(),
                                   [&](const auto &h) { return h.get() == he->twin; });
            if (it != _half_edges.end())
                twin_idx = std::distance(_half_edges.begin(), it);
        }
        bool deleted = he->deleted;
        std::cout << from << " " << to << " " << twin_idx << " " << deleted << "\n";
    }
}
