#include "halfedge.h"
#include <iostream>

uint64_t key(uint32_t vertexIndex1, uint32_t vertexIndex2)
{
    // Create a unique key for the edge defined by vertexIndex1 and vertexIndex2
    if (vertexIndex1 > vertexIndex2)
    {
        // Ensure the smaller index is always first to maintain consistency
        std::swap(vertexIndex1, vertexIndex2);
    }

    return (static_cast<uint64_t>(vertexIndex1) << 32) | vertexIndex2;
}

void HalfEdgeRepr::initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                                   const std::vector<Eigen::Vector3i> &faces)
{
    clear();
    std::unordered_map<uint64_t, HalfEdge *> edgeMap;
    auto processEdge = [&](int from, int to, HalfEdge *h)
    {
        uint64_t twinKey = key(from, to);

        if (edgeMap.find(twinKey) != edgeMap.end())
        {
            // Twin half-edge exists, link them
            HalfEdge *twin = edgeMap[twinKey];
            h->twin = twin;
            twin->twin = h;

            h->vertex = _vertices[(h->twin->vertex->id == from) ? to : from].get();

            // Create a new edge now that we have both half-edges
            _edges.push_back(std::make_unique<Edge>(_edges.size(), h));

            // Link the edge to both half-edges
            h->edge = _edges.back().get();
            twin->edge = _edges.back().get();
        }
        else
        {
            h->vertex = _vertices[from].get();
            edgeMap[key(from, to)] = h;
        }
    };

    for (int i = 0; i < vertices.size(); ++i)
    {
        _vertices.push_back(std::make_unique<Vertex>(i, vertices[i]));
    }

    for (size_t i = 0; i < faces.size(); i++)
    {
        auto face = std::make_unique<Face>(_faces.size());
        auto he1 = std::make_unique<HalfEdge>(_half_edges.size());
        auto he2 = std::make_unique<HalfEdge>(_half_edges.size() + 1);
        auto he3 = std::make_unique<HalfEdge>(_half_edges.size() + 2);

        he1->face = face.get();
        he2->face = face.get();
        he3->face = face.get();

        // Sets vertex, twin, and edge
        processEdge(faces[i][0], faces[i][1], he1.get());
        processEdge(faces[i][1], faces[i][2], he2.get());
        processEdge(faces[i][2], faces[i][0], he3.get());

        // Set next
        if (he1->vertex->id == faces[i][0] && he2->vertex->id == faces[i][1] &&
            he3->vertex->id == faces[i][2])
        {
            he1->next = he2.get();
            he2->next = he3.get();
            he3->next = he1.get();
        }
        else if (he1->vertex->id == faces[i][1] && he2->vertex->id == faces[i][2] &&
                 he3->vertex->id == faces[i][0])
        {
            he1->next = he3.get();
            he2->next = he1.get();
            he3->next = he2.get();
        }
        else
        {
            throw std::runtime_error(
                "Half-edge vertex assignment does not match face vertex indices");
        }

        // Overwrite so that vertex points to most recently processed half-edge
        he1->vertex->halfEdge = he1.get();
        he2->vertex->halfEdge = he2.get();
        he3->vertex->halfEdge = he3.get();

        // Overwrite face's half-edge pointer to point to the most recently processed half-edge
        face->halfEdge = he3.get();

        _faces.push_back(std::move(face));
        _half_edges.push_back(std::move(he1));
        _half_edges.push_back(std::move(he2));
        _half_edges.push_back(std::move(he3));
    }
}

void HalfEdgeRepr::outputToVectors(std::vector<Eigen::Vector3f> &vertices,
                                   std::vector<Eigen::Vector3i> &faces)
{
    sweep();

    vertices.clear();
    faces.clear();

    for (const auto &vertex : _vertices)
    {
        vertices.push_back(vertex->position);
    }

    for (const auto &face : _faces)
    {
        Eigen::Vector3i faceIndices;
        HalfEdge *he = face->halfEdge;
        for (int i = 0; i < 3; ++i)
        {
            faceIndices[i] = he->vertex->id;
            he = he->next;
        }
        faces.push_back(faceIndices);
    }
}

void HalfEdgeRepr::he_denoise(int numIterations, double sigma_c, double sigma_s, double rho)
{
    sweep();
    validate();
}
