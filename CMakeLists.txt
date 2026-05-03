cmake_minimum_required(VERSION 3.16)

# Sets project name
project(simulation LANGUAGES CXX C)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Sets C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specifies required Qt components
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Concurrent)
find_package(Qt6 REQUIRED COMPONENTS Xml)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(Qt6 REQUIRED COMPONENTS OpenGL)
find_package(Qt6 REQUIRED COMPONENTS OpenGLWidgets)
find_package(Qt6 REQUIRED COMPONENTS Gui)

if(NOT APPLE)
    find_package(OpenMP REQUIRED)
endif()

# Optional: CHOLMOD for faster sparse Cholesky (paper uses this).
find_path(CHOLMOD_INCLUDE_DIR cholmod.h PATHS /usr/include/suitesparse /usr/local/include)
find_library(CHOLMOD_LIBRARY cholmod)
if (CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARY)
    message(STATUS "Found CHOLMOD: ${CHOLMOD_LIBRARY}")
    add_compile_definitions(HAVE_CHOLMOD)
else()
    message(STATUS "CHOLMOD not found, using Eigen's SimplicialLDLT (slower)")
endif()

# Allows you to include files from within those directories, without prefixing their filepaths
include_directories(src)
include_directories(libs)

# Specifies .cpp and .h files to be passed to the compiler
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/mainwindow.cpp
    src/simulation.cpp
    src/shell_mesh.cpp
    src/geometry.cpp
    src/energy.cpp
    src/implicit_euler.cpp
    src/diffusion.cpp
    src/rest_metric.cpp
    src/glwidget.cpp
    src/graphics/camera.cpp
    src/graphics/graphicsdebug.cpp
    src/graphics/meshloader.cpp
    src/graphics/shader.cpp
    src/graphics/shape.cpp

    src/mainwindow.h
    src/simulation.h
    src/shell_mesh.h
    src/shell_energy.h
    src/geometry.h
    src/energy.h
    src/implicit_euler.h
    src/diffusion.h
    src/display_mode.h
    src/rest_metric.h
    src/glwidget.h
    src/graphics/camera.h
    src/graphics/graphicsdebug.h
    src/graphics/meshloader.h
    src/graphics/shader.h
    src/graphics/shape.h

    util/tiny_obj_loader.h
    util/unsupportedeigenthing/OpenGLSupport
)

# GLEW: this creates its library and allows you to `#include "GL/glew.h"`
add_library(StaticGLEW STATIC glew/src/glew.c)
include_directories(${PROJECT_NAME} PRIVATE glew/include)

# Specifies libraries to be linked (Qt components, glew, etc)
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::OpenGL
    Qt::OpenGLWidgets
    Qt::Widgets
    Qt::Xml
    StaticGLEW
    OpenMP::OpenMP_CXX
)

if (CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARY)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CHOLMOD_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CHOLMOD_LIBRARY})
endif()

# This allows you to `#include "Eigen/..."`
target_include_directories(${PROJECT_NAME} PRIVATE
    extern
)

# Specifies other files
qt6_add_resources(${PROJECT_NAME} "Resources"
    PREFIX
        "/"
    FILES
        resources/shaders/shader.frag
        resources/shaders/shader.vert
        resources/shaders/skybox.frag
        resources/shaders/skybox.vert
)

# GLEW: this provides support for Windows (including 64-bit)
if (WIN32)
  add_compile_definitions(GLEW_STATIC)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    opengl32
    glu32
  )
endif()

# Set this flag to silence warnings on Windows
if (MSVC OR MSYS OR MINGW)
  set(CMAKE_CXX_FLAGS "-Wno-volatile")
endif()

# Set this flag to silence warnings on MacOS
if (APPLE)
  set(CMAKE_CXX_FLAGS "-Wno-deprecated-volatile")
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
  # 2. Now ask CMake to find it naturally
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()


