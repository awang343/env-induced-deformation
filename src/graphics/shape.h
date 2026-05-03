#ifndef SHAPE_H
#define SHAPE_H

#include <GL/glew.h>
#include <vector>

#include <Eigen/Dense>

class Shader;

class Shape
{
public:
    Shape();

    void cleanup();

    void init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals, const std::vector<Eigen::Vector3i> &triangles);
    void init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3i> &triangles);

    void setVertices(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals);
    void setVertices(const std::vector<Eigen::Vector3d> &vertices);

    // Per-face data for heatmap visualization.
    // Two values per face (e.g., energy+0, or m_plus+m_minus).
    void setFaceData(const std::vector<double> &channel0,
                     const std::vector<double> &channel1);

    void setModelMatrix(const Eigen::Affine3f &model);

    // Cycles display modes. numModes controls how many are available.
    void cycleDisplayMode(int numModes = 3);

    void draw(Shader *shader);

    // 0 = solid, 1 = energy heatmap, 2 = moisture heatmap, 3 = wireframe
    int displayMode() const { return m_displayMode; }
    static const char* displayModeName(int mode);

private:
    GLuint m_surfaceVao;
    GLuint m_surfaceVbo;
    GLuint m_surfaceIbo;
    GLuint m_energyVbo;  // per-vertex energy attribute

    unsigned int m_numSurfaceVertices;
    unsigned int m_verticesSize;
    float m_red;
    float m_blue;
    float m_green;
    float m_alpha;

    std::vector<Eigen::Vector3i> m_faces;

    Eigen::Matrix4f m_modelMatrix;

    int m_displayMode = 0;  // 0=solid, 1=heatmap, 2=wireframe
};

#endif // SHAPE_H
