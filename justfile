# Variables
SRC_DIR := "."
BUILD_DIR := "build"
EXECUTABLE := "simulation"
BUILD_TYPE := "Release"   # default build type

alias b := build
alias r := run

# Configure CMake
configure:
    cmake -S {{SRC_DIR}} -B {{BUILD_DIR}} \
        -DCMAKE_BUILD_TYPE={{BUILD_TYPE}} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    ln -sfn ./{{BUILD_DIR}}/compile_commands.json compile_commands.json

# Build the project
build: configure
    cmake --build {{BUILD_DIR}}

run config: build
    ./{{BUILD_DIR}}/{{EXECUTABLE}} {{config}}

# Clean build directory
clean:
    rm -rf {{BUILD_DIR}}
