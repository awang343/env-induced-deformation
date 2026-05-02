#pragma once

#include "shell_mesh.h"
#include "energy.h"
#include <vector>

// Diffuse m_plus and m_minus by one timestep using implicit integration:
//   (M_G + dt*D*K_G) * m_new = M_G * (m_old + dt*s)
// M_G: Galerkin mass matrix on prisms F x [-h/2, h/2]
// K_G: Laplace-Beltrami stiffness (in-plane + through-thickness)
// D: diffusion coefficient
// s: source term (per-vertex, top and bottom)
void diffuseMoisture(const ShellMesh &mesh,
                     const ShellRestState &rest,
                     const MaterialParams &mat,
                     double dt, double diffusivity,
                     std::vector<double> &mPlus,
                     std::vector<double> &mMinus,
                     const std::vector<double> &sPlus = {},
                     const std::vector<double> &sMinus = {});
