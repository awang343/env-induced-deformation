#include "Eigen/Dense"
#include <deque>
#include <memory>

struct HalfEdge
{
    int id = -1;
    HalfEdge *twin;
    HalfEdge *next;
    struct Vertex *vertex;
    struct Edge *edge;
    struct Face *face;
    bool deleted;

    HalfEdge(int id)
        : id(id), twin(nullptr), next(nullptr), vertex(nullptr), edge(nullptr), face(nullptr),
          deleted(false)
    {
    }
};

struct Vertex
{
    int id = -1;
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    HalfEdge *halfEdge = nullptr;
    bool deleted = false;

    Eigen::Matrix4f quadric =
        Eigen::Matrix4f::Zero(); // For quadric error metric simplification only
    Eigen::Vector3f normal = Eigen::Vector3f::Zero(); // For remeshing only

    Vertex(int id, const Eigen::Vector3f &pos)
        : id(id), position(pos), halfEdge(nullptr), quadric(Eigen::Matrix4f::Zero()), deleted(false)
    {
    }
};

struct Edge
{
    int id = -1;
    HalfEdge *halfEdge = nullptr;
    bool deleted = false;

    bool is_new = false; // For subdivision only
    float length = 0.0f; // For remeshing only

    Edge(int id) : id(id), halfEdge(nullptr), is_new(false), deleted(false) {}
    Edge(int id, HalfEdge *he) : id(id), halfEdge(he), is_new(false), deleted(false) {}
    Edge(int id, HalfEdge *he, bool is_new) : id(id), halfEdge(he), is_new(is_new), deleted(false)
    {
    }
};

struct Face
{
    int id = -1;
    HalfEdge *halfEdge = nullptr;
    bool deleted = false;

    Eigen::Vector4f plane = Eigen::Vector4f::Zero(); // For quadric error metric simplification only
    Eigen::Vector3f normal = Eigen::Vector3f::Zero(); // For remeshing only

    Face(int id) : id(id), halfEdge(nullptr), deleted(false) {}
};

class HalfEdgeRepr
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    void initFromVectors(const std::vector<Eigen::Vector3f> &vertices,
                         const std::vector<Eigen::Vector3i> &faces);
    void outputToVectors(std::vector<Eigen::Vector3f> &vertices,
                         std::vector<Eigen::Vector3i> &faces);

    void loop_subdivision();
    void quadric_simplification(int numFacesToRemove);
    void isotropic_remesh(int numIterations, double tangentialSmoothingWeight);
    void he_denoise(int numIterations, double sigma_c, double sigma_s, double rho);

  private:
    void sweep();
    void clear();
    void validate();
    void dump();
    int degree(Vertex *v);

    void edgeFlip(HalfEdge *he);
    void edgeSplit(HalfEdge *he);
    void edgeCollapse(HalfEdge *he);

    bool edgeFlipValid(HalfEdge *he);
    bool edgeCollapseValid(HalfEdge *he);

    std::deque<std::unique_ptr<HalfEdge>> _half_edges;
    std::deque<std::unique_ptr<Vertex>> _vertices;
    std::deque<std::unique_ptr<Face>> _faces;
    std::deque<std::unique_ptr<Edge>> _edges;
};
