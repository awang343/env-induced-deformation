#include "geometry.h"
#include <cmath>

using namespace Eigen;

Matrix3d skew(const Vector3d &v)
{
    Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

double svNormSq(const Matrix2d &M, double alpha, double beta)
{
    double tr  = M.trace();
    double tr2 = (M * M).trace();
    return 0.5 * alpha * tr * tr + beta * tr2;
}

template<int N>
Matrix<double,N,N> projectPSD(const Matrix<double,N,N> &H)
{
    SelfAdjointEigenSolver<Matrix<double,N,N>> es(H);
    auto vals = es.eigenvalues().cwiseMax(0.0);
    return es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
}
// Explicit instantiation for the sizes we use.
template Matrix<double,9,9> projectPSD<9>(const Matrix<double,9,9>&);
template Matrix<double,18,18> projectPSD<18>(const Matrix<double,18,18>&);

void computeFaceNormals(const ShellMesh &mesh, std::vector<Vector3d> &normals)
{
    const int nF = mesh.numFaces();
    normals.resize(nF);
    for (int f = 0; f < nF; ++f) {
        const auto &t = mesh.faces[f];
        normals[f] = (mesh.vertices[t[1]] - mesh.vertices[t[0]])
                     .cross(mesh.vertices[t[2]] - mesh.vertices[t[0]]);
    }
}

Matrix2d firstFundamentalForm(const ShellMesh &mesh, int face)
{
    const Vector3i &tri = mesh.faces[face];
    Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
    Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];
    double d12 = e1.dot(e2);
    Matrix2d a;
    a << e1.dot(e1), d12, d12, e2.dot(e2);
    return a;
}

Matrix<double, 4, 9> firstFFDeriv(const ShellMesh &mesh, int face)
{
    const Vector3i &tri = mesh.faces[face];
    Vector3d e1 = mesh.vertices[tri[1]] - mesh.vertices[tri[0]];
    Vector3d e2 = mesh.vertices[tri[2]] - mesh.vertices[tri[0]];
    Matrix<double, 4, 9> D = Matrix<double, 4, 9>::Zero();
    D.block<1,3>(0, 0) = -2.0 * e1.transpose();
    D.block<1,3>(0, 3) =  2.0 * e1.transpose();
    D.block<1,3>(1, 0) = -(e1 + e2).transpose();
    D.block<1,3>(1, 3) =  e2.transpose();
    D.block<1,3>(1, 6) =  e1.transpose();
    D.row(2) = D.row(1);
    D.block<1,3>(3, 0) = -2.0 * e2.transpose();
    D.block<1,3>(3, 6) =  2.0 * e2.transpose();
    return D;
}

void firstFFHessian(Matrix<double, 9, 9> ahess[4])
{
    for (int i = 0; i < 4; ++i) ahess[i].setZero();
    Matrix3d I3 = Matrix3d::Identity();
    ahess[0].block<3,3>(0,0) =  2*I3; ahess[0].block<3,3>(3,3) =  2*I3;
    ahess[0].block<3,3>(0,3) = -2*I3; ahess[0].block<3,3>(3,0) = -2*I3;
    ahess[1].block<3,3>(0,0) =  2*I3; ahess[1].block<3,3>(0,3) = -I3;
    ahess[1].block<3,3>(0,6) = -I3;   ahess[1].block<3,3>(3,0) = -I3;
    ahess[1].block<3,3>(3,6) =  I3;   ahess[1].block<3,3>(6,0) = -I3;
    ahess[1].block<3,3>(6,3) =  I3;
    ahess[2] = ahess[1];
    ahess[3].block<3,3>(0,0) =  2*I3; ahess[3].block<3,3>(6,6) =  2*I3;
    ahess[3].block<3,3>(0,6) = -2*I3; ahess[3].block<3,3>(6,0) = -2*I3;
}

Matrix2d secondFundamentalForm(const ShellMesh &mesh,
                               const std::vector<Vector3d> &faceNormals, int face)
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d nc = faceNormals[face];
    double II[3];
    for (int i = 0; i < 3; ++i) {
        int ip1 = (i+1)%3, ip2 = (i+2)%3;
        Vector3d qv = q[tri[ip1]] + q[tri[ip2]] - 2.0*q[tri[i]];
        int eid = mesh.faceEdges[face][i];
        const auto &r = mesh.edgeFaces[eid];
        int of = (r[0].face == face) ? r[1].face : r[0].face;
        if (of == -1) { II[i] = 0; continue; }
        Vector3d mv = faceNormals[of] + nc;
        double mn = mv.norm();
        II[i] = (mn > 0) ? qv.dot(faceNormals[of]) / mn : 0;
    }
    Matrix2d b;
    b << II[0]+II[1], II[0], II[0], II[0]+II[2];
    return b;
}

Matrix<double, 4, 18> secondFFDeriv(
    const ShellMesh &mesh, const std::vector<Vector3d> &faceNormals,
    int face, int oppVerts[3])
{
    const Vector3i &tri = mesh.faces[face];
    const Vector3d *q = mesh.vertices.data();
    Vector3d nc = faceNormals[face];
    Vector3d e1 = q[tri[1]] - q[tri[0]], e2 = q[tri[2]] - q[tri[0]];

    Matrix<double,3,9> dnc = Matrix<double,3,9>::Zero();
    dnc.block<3,3>(0,0) = skew(q[tri[2]]-q[tri[1]]);
    dnc.block<3,3>(0,3) = -skew(e2);
    dnc.block<3,3>(0,6) = skew(e1);

    Matrix<double,1,18> dII[3];
    for (int i = 0; i < 3; ++i) dII[i].setZero();

    for (int i = 0; i < 3; ++i) {
        int ip1=(i+1)%3, ip2=(i+2)%3;
        Vector3d qv = q[tri[ip1]] + q[tri[ip2]] - 2.0*q[tri[i]];
        int eid = mesh.faceEdges[face][i];
        const auto &r = mesh.edgeFaces[eid];
        int of = (r[0].face==face) ? r[1].face : r[0].face;
        int olv = (r[0].face==face) ? r[1].localOppVtx : r[0].localOppVtx;
        if (of == -1) { oppVerts[i] = -1; continue; }
        oppVerts[i] = mesh.faces[of][olv];

        Vector3d no = faceNormals[of], mv = no + nc;
        double mn = mv.norm();
        if (mn < 1e-16) continue;
        double IIv = qv.dot(no) / mn;

        Vector3d nom = no / mn;
        dII[i].segment<3>(3*i)   += -2.0 * nom.transpose();
        dII[i].segment<3>(3*ip1) +=        nom.transpose();
        dII[i].segment<3>(3*ip2) +=        nom.transpose();

        int ov0=mesh.faces[of][olv], ov1=mesh.faces[of][(olv+1)%3], ov2=mesh.faces[of][(olv+2)%3];
        Vector3d oe1=q[ov1]-q[ov0], oe2=q[ov2]-q[ov0];
        Matrix3d dno0=skew(q[ov2]-q[ov1]), dno1=-skew(oe2), dno2=skew(oe1);

        auto findL = [&](int gv) { for(int j=0;j<3;++j) if(tri[j]==gv) return j; return -1; };
        int l1=findL(ov1), l2=findL(ov2);

        Vector3d qom = qv / mn;
        dII[i].segment<3>(9+3*i) += qom.transpose() * dno0;
        if (l1>=0) dII[i].segment<3>(3*l1) += qom.transpose() * dno1;
        if (l2>=0) dII[i].segment<3>(3*l2) += qom.transpose() * dno2;

        double c = -IIv / (mn*mn);
        RowVector3d mt = mv.transpose();
        dII[i].segment<3>(9+3*i) += c * mt * dno0;
        if (l1>=0) dII[i].segment<3>(3*l1) += c * mt * dno1;
        if (l2>=0) dII[i].segment<3>(3*l2) += c * mt * dno2;
        for (int j=0; j<3; ++j)
            dII[i].segment<3>(3*j) += c * mt * dnc.block<3,3>(0,3*j);
    }

    Matrix<double,4,18> bD;
    bD.row(0) = dII[0]+dII[1];
    bD.row(1) = dII[0];
    bD.row(2) = dII[0];
    bD.row(3) = dII[0]+dII[2];
    return bD;
}
