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

    // Per-face energy values for heatmap visualization.
    // One value per face — gets expanded to 3 identical values per triangle vertex.
    void setFaceEnergies(const std::vector<double> &energies);

    void setModelMatrix(const Eigen::Affine3f &model);

    // Cycles: solid → heatmap → wireframe → solid
    void cycleDisplayMode();

    void draw(Shader *shader);

    // 0 = solid color, 1 = energy heatmap, 2 = wireframe
    int displayMode() const { return m_displayMode; }

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
