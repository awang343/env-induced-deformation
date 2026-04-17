#include "glwidget.h"

#include <QApplication>
#include <QKeyEvent>
#include <QPainter>
#include <iostream>

#define SPEED 1.5
#define ROTATE_SPEED 0.0025

using namespace std;

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent), m_deltaTimeProvider(), m_intervalTimer(), m_sim(), m_camera(),
      m_shader(), m_forward(), m_sideways(), m_vertical(), m_lastX(), m_lastY(), m_capture(false)
{
    // GLWidget needs all mouse move events, not just mouse drag events
    setMouseTracking(true);

    // Hide the cursor since this is a fullscreen app
    QApplication::setOverrideCursor(Qt::ArrowCursor);

    // GLWidget needs keyboard focus
    setFocusPolicy(Qt::StrongFocus);

    // Function tick() will be called once per interval
    connect(&m_intervalTimer, SIGNAL(timeout()), this, SLOT(tick()));
}

GLWidget::~GLWidget()
{
    if (m_shader != nullptr)
        delete m_shader;
    if (m_skyboxShader != nullptr)
        delete m_skyboxShader;
}

// ================== Basic OpenGL Overrides

void GLWidget::initializeGL()
{
    // Initialize GL extension wrangler
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
        fprintf(stderr, "Error while initializing GLEW: %s\n", glewGetErrorString(err));
    fprintf(stdout, "Successfully initialized GLEW %s\n", glewGetString(GLEW_VERSION));

    // Set clear color to white
    glClearColor(1, 1, 1, 1);

    // Enable depth-testing. Back-face culling stays off because thin
    // shells have two visible sides; the fragment shader flips the
    // normal for back-facing fragments so both sides are lit.
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // Initialize shaders
    m_shader = new Shader(":/resources/shaders/shader.vert", ":/resources/shaders/shader.frag");
    m_skyboxShader = new Shader(":/resources/shaders/skybox.vert", ":/resources/shaders/skybox.frag");

    // Fullscreen quad for skybox
    float quadVerts[] = {
        -1.f, -1.f,
         1.f, -1.f,
        -1.f,  1.f,
         1.f,  1.f,
    };
    glGenVertexArrays(1, &m_skyboxVao);
    glGenBuffers(1, &m_skyboxVbo);
    glBindVertexArray(m_skyboxVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_skyboxVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    m_sim.init(mesh_path);

    // Initialize camera with a reasonable transform
    Eigen::Vector3f eye = {0, 2, -8};
    Eigen::Vector3f target = {-1.2, 1, 0};
    m_camera.lookAt(eye, target);
    m_camera.setOrbitPoint(target);
    m_camera.setPerspective(120, width() / static_cast<float>(height()), 0.1, 50);

    m_deltaTimeProvider.start();
    m_intervalTimer.start(1000 / 60);
}

void GLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw skybox (behind everything)
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    m_skyboxShader->bind();
    Eigen::Matrix4f projView = m_camera.getProjection() * m_camera.getView();
    Eigen::Matrix4f invProjView = projView.inverse();
    m_skyboxShader->setUniform("invProjView", invProjView);
    Eigen::Matrix4f invView = m_camera.getView().inverse();
    Eigen::Vector3f eye = invView.block<3,1>(0,3);
    m_skyboxShader->setUniform("camPos", eye);
    glBindVertexArray(m_skyboxVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    m_skyboxShader->unbind();
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // Draw scene
    m_shader->bind();
    m_shader->setUniform("proj", m_camera.getProjection());
    m_shader->setUniform("view", m_camera.getView());
    m_sim.draw(m_shader);
    m_shader->unbind();

    // 2D text overlay via QPainter
    QPainter painter(this);
    painter.setPen(Qt::black);
    painter.setFont(QFont("Monospace", 14));
    painter.drawText(10, 24, QString("Step %1").arg(m_frameCount));
    painter.end();
}

void GLWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    m_camera.setAspect(static_cast<float>(w) / h);
}

void GLWidget::setMeshPath(const QString &meshPath)
{
    mesh_path = meshPath.toStdString();
}

// ================== Event Listeners

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    m_capture = true;
    m_lastX = event->position().x();
    m_lastY = event->position().y();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (!m_capture)
        return;

    int currX = event->position().x();
    int currY = event->position().y();

    int deltaX = currX - m_lastX;
    int deltaY = currY - m_lastY;

    if (deltaX == 0 && deltaY == 0)
        return;

    m_camera.rotate(deltaY * ROTATE_SPEED, -deltaX * ROTATE_SPEED);

    m_lastX = currX;
    m_lastY = currY;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    m_capture = false;
}

void GLWidget::wheelEvent(QWheelEvent *event)
{
    float zoom = 1 - event->pixelDelta().y() * 0.1f / 120.f;
    m_camera.zoom(zoom);
}

void GLWidget::keyPressEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
        return;

    switch (event->key())
    {
    case Qt::Key_W:
        m_forward += SPEED;
        break;
    case Qt::Key_S:
        m_forward -= SPEED;
        break;
    case Qt::Key_A:
        m_sideways -= SPEED;
        break;
    case Qt::Key_D:
        m_sideways += SPEED;
        break;
    case Qt::Key_Q:
        m_vertical -= SPEED;
        break;
    case Qt::Key_E:
        m_vertical += SPEED;
        break;
    case Qt::Key_C:
        m_camera.toggleIsOrbiting();
        break;
    case Qt::Key_T:
        m_sim.toggleWire();
        break;
    case Qt::Key_Space:
        m_sim.togglePause();
        break;
    case Qt::Key_R:
    case Qt::Key_P:
        m_sim.reset();
        m_frameCount = 0;
        break;
    case Qt::Key_O:
        m_sim.toggleParallel();
        break;
    case Qt::Key_G:
        m_sim.cycleGrowthDemo();
        break;
    case Qt::Key_BracketRight:
        m_physicsRate = std::max(1, m_physicsRate / 2);
        std::cout << "Physics: every " << m_physicsRate << " frame(s)" << std::endl;
        break;
    case Qt::Key_BracketLeft:
        m_physicsRate = std::min(128, m_physicsRate * 2);
        std::cout << "Physics: every " << m_physicsRate << " frame(s)" << std::endl;
        break;
    case Qt::Key_Period:
        m_sim.singleStep();
        m_frameCount++;
        update();
        break;
    case Qt::Key_Escape:
        QApplication::quit();
    }
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat())
        return;

    switch (event->key())
    {
    case Qt::Key_W:
        m_forward -= SPEED;
        break;
    case Qt::Key_S:
        m_forward += SPEED;
        break;
    case Qt::Key_A:
        m_sideways += SPEED;
        break;
    case Qt::Key_D:
        m_sideways -= SPEED;
        break;
    case Qt::Key_Q:
        m_vertical += SPEED;
        break;
    case Qt::Key_E:
        m_vertical -= SPEED;
        break;
    }
}

// ================== Physics Tick

void GLWidget::tick()
{
    float deltaSeconds = m_deltaTimeProvider.restart() / 1000.f;

    m_tickCount++;
    if (m_tickCount >= m_physicsRate) {
        m_tickCount = 0;
        m_sim.update(deltaSeconds);
        // Only increment if update actually stepped (not paused/empty).
        if (!m_sim.isPaused()) m_frameCount++;
    }

    // Move camera
    auto look = m_camera.getLook();
    look.y() = 0;
    look.normalize();
    Eigen::Vector3f perp(-look.z(), 0, look.x());
    Eigen::Vector3f moveVec = m_forward * look.normalized() + m_sideways * perp.normalized() +
                              m_vertical * Eigen::Vector3f::UnitY();
    moveVec *= deltaSeconds;
    m_camera.move(moveVec);

    update();
}
