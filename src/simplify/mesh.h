#pragma once

#include <vector>

#include "Eigen/Dense"
#include "Eigen/StdVector"
#include "halfedge/halfedge.h"

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3f);
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3i);

class Mesh
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                         const std::vector<Eigen::Vector3i> &faces);

    void simplify(int numFacesToRemove);

    const std::vector<Eigen::Vector3f>& vertices() const { return _vertices; }
    const std::vector<Eigen::Vector3i>& faces() const { return _faces; }

  private:
    std::vector<Eigen::Vector3f> _vertices;
    std::vector<Eigen::Vector3i> _faces;

    HalfEdgeRepr _halfEdgeRepr;
    void toHalfEdge();
    void fromHalfEdge();
};
