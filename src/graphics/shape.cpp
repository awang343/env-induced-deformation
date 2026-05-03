#include "shape.h"

#include <iostream>

#include "graphics/shader.h"

using namespace Eigen;

Shape::Shape()
    : m_surfaceVao(-1),
      m_surfaceVbo(-1),
      m_surfaceIbo(-1),
      m_energyVbo(-1),
      m_numSurfaceVertices(),
      m_verticesSize(),
      m_red(0.55f),
      m_blue(0.95f),
      m_green(0.35f),
      m_alpha(1.f),
      m_modelMatrix(Eigen::Matrix4f::Identity()),
      m_displayMode(0)
{
}

void Shape::cleanup()
{
    if (m_surfaceVao != static_cast<GLuint>(-1)) {
        glDeleteVertexArrays(1, &m_surfaceVao);
        glDeleteBuffers(1, &m_surfaceVbo);
        glDeleteBuffers(1, &m_surfaceIbo);
        if (m_energyVbo != static_cast<GLuint>(-1)) glDeleteBuffers(1, &m_energyVbo);
        m_surfaceVao = -1;
        m_energyVbo = -1;
    }
    m_numSurfaceVertices = 0;
    m_verticesSize = 0;
}

void Shape::init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals, const std::vector<Eigen::Vector3i> &triangles)
{
    cleanup();
    if(vertices.size() != normals.size()) {
        std::cerr << "Vertices and normals are not the same size" << std::endl;
        return;
    }
    glGenBuffers(1, &m_surfaceVbo);
    glGenBuffers(1, &m_surfaceIbo);
    glGenBuffers(1, &m_energyVbo);
    glGenVertexArrays(1, &m_surfaceVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3 * 2, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * vertices.size() * 3, static_cast<const void *>(vertices.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3, sizeof(double) * vertices.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Energy VBO: one float per expanded vertex (3 per face), initialized to 0.
    std::vector<float> zeroData(triangles.size() * 3 * 2, 0.0f);
    glBindBuffer(GL_ARRAY_BUFFER, m_energyVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * zeroData.size(), zeroData.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * triangles.size(), static_cast<const void *>(triangles.data()), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(m_surfaceVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, reinterpret_cast<GLvoid *>(sizeof(double) * vertices.size() * 3));
    glBindBuffer(GL_ARRAY_BUFFER, m_energyVbo);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_numSurfaceVertices = triangles.size() * 3;
    m_verticesSize = vertices.size();
    m_faces = triangles;
}

void Shape::init(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3i> &triangles)
{
    cleanup();
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3i> faces;
    verts.reserve(triangles.size() * 3);
    normals.reserve(triangles.size() * 3);
    for(auto& f : triangles) {
        auto& v1 = vertices[f[0]];
        auto& v2 = vertices[f[1]];
        auto& v3 = vertices[f[2]];
        auto e1 = v2 - v1;
        auto e2 = v3 - v1;
        auto n = e1.cross(e2);
        int s = verts.size();
        faces.push_back(Eigen::Vector3i(s, s + 1, s + 2));
        normals.push_back(n);
        normals.push_back(n);
        normals.push_back(n);
        verts.push_back(v1);
        verts.push_back(v2);
        verts.push_back(v3);
    }
    glGenBuffers(1, &m_surfaceVbo);
    glGenBuffers(1, &m_surfaceIbo);
    glGenBuffers(1, &m_energyVbo);
    glGenVertexArrays(1, &m_surfaceVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3 * 2, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * verts.size() * 3, static_cast<const void *>(verts.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3, sizeof(double) * verts.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::vector<float> zeroData2(faces.size() * 3 * 2, 0.0f);
    glBindBuffer(GL_ARRAY_BUFFER, m_energyVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * zeroData2.size(), zeroData2.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * 3 * faces.size(), static_cast<const void *>(faces.data()), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(m_surfaceVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, reinterpret_cast<GLvoid *>(sizeof(double) * verts.size() * 3));
    glBindBuffer(GL_ARRAY_BUFFER, m_energyVbo);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLvoid *>(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_surfaceIbo);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    m_numSurfaceVertices = faces.size() * 3;
    m_verticesSize = vertices.size();
    m_faces = triangles;

    m_red = 0.55f;
    m_green = 0.35f;
    m_blue = 0.95f;
    m_alpha = 1.f;
}

void Shape::setVertices(const std::vector<Eigen::Vector3d> &vertices)
{
    if(vertices.size() != m_verticesSize) {
        std::cerr << "You can't set vertices to a vector that is a different length that what shape was inited with" << std::endl;
        return;
    }
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3d> normals;
    verts.reserve(m_faces.size() * 3);
    normals.reserve(m_faces.size() * 3);
    for(auto& f : m_faces) {
        auto& v1 = vertices[f[0]];
        auto& v2 = vertices[f[1]];
        auto& v3 = vertices[f[2]];
        auto e1 = v2 - v1;
        auto e2 = v3 - v1;
        auto n = e1.cross(e2);
        normals.push_back(n);
        normals.push_back(n);
        normals.push_back(n);
        verts.push_back(v1);
        verts.push_back(v2);
        verts.push_back(v3);
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * verts.size() * 3, static_cast<const void *>(verts.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * verts.size() * 3, sizeof(double) * verts.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::setFaceData(const std::vector<double> &ch0,
                        const std::vector<double> &ch1)
{
    // Expand per-face (ch0, ch1) to per-expanded-vertex vec2 (3 per face).
    std::vector<float> expanded;
    expanded.reserve(m_faces.size() * 3 * 2);
    for (size_t f = 0; f < m_faces.size(); ++f) {
        float a = (f < ch0.size()) ? static_cast<float>(ch0[f]) : 0.0f;
        float b = (f < ch1.size()) ? static_cast<float>(ch1[f]) : 0.0f;
        for (int v = 0; v < 3; ++v) {
            expanded.push_back(a);
            expanded.push_back(b);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_energyVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * expanded.size(), expanded.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::setModelMatrix(const Eigen::Affine3f &model)
{
    m_modelMatrix = model.matrix();
}

void Shape::cycleDisplayMode(int numModes)
{
    m_displayMode = (m_displayMode + 1) % numModes;
    std::cout << "Display: " << displayModeName(m_displayMode) << std::endl;
}

const char* Shape::displayModeName(int mode)
{
    static const char *names[] = {"Solid", "Energy", "Moisture", "Wireframe"};
    if (mode >= 0 && mode < 4) return names[mode];
    return "Unknown";
}

void Shape::setVertices(const std::vector<Eigen::Vector3d> &vertices, const std::vector<Eigen::Vector3d> &normals)
{
    if(vertices.size() != normals.size()) {
        std::cerr << "Vertices and normals are not the same size" << std::endl;
        return;
    }
    if(vertices.size() != m_verticesSize) {
        std::cerr << "You can't set vertices to a vector that is a different length that what shape was inited with" << std::endl;
        return;
    }
    glBindBuffer(GL_ARRAY_BUFFER, m_surfaceVbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(double) * vertices.size() * 3, static_cast<const void *>(vertices.data()));
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(double) * vertices.size() * 3, sizeof(double) * vertices.size() * 3, static_cast<const void *>(normals.data()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Shape::draw(Shader *shader)
{
    Eigen::Matrix3f m3 = m_modelMatrix.topLeftCorner(3, 3);
    Eigen::Matrix3f inverseTransposeModel = m3.inverse().transpose();

    shader->setUniform("displayMode", m_displayMode);
    shader->setUniform("model", m_modelMatrix);
    shader->setUniform("inverseTransposeModel", inverseTransposeModel);
    shader->setUniform("red",   m_red);
    shader->setUniform("green", m_green);
    shader->setUniform("blue",  m_blue);
    shader->setUniform("alpha", m_alpha);
    glBindVertexArray(m_surfaceVao);
    if (m_displayMode == 3) {  // wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, m_numSurfaceVertices, GL_UNSIGNED_INT, reinterpret_cast<GLvoid *>(0));
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glDrawElements(GL_TRIANGLES, m_numSurfaceVertices, GL_UNSIGNED_INT, reinterpret_cast<GLvoid *>(0));
    }
    glBindVertexArray(0);
}
