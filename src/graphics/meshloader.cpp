#include "graphics/meshloader.h"

#include <iostream>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>

using namespace Eigen;

bool MeshLoader::loadTriMesh(const std::string &filepath,
                             std::vector<Eigen::Vector3d> &vertices,
                             std::vector<Eigen::Vector3i> &faces)
{
    QString qpath = QString::fromStdString(filepath);
    QFile file(qpath);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        std::cout << "Error opening file: " << filepath << std::endl;
        return false;
    }
    QTextStream in(&file);

    // Matches Wavefront-OBJ style lines:
    //   v  x y z                    (possibly with scientific notation)
    //   f  i  j  k                  (1-indexed vertex ids)
    //   f  i/t/n  j/t/n  k/t/n      (only the vertex index is used)
    // Leading `vn`/`vt` lines and comments are ignored.
    const QString num = "(-?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)";
    QRegularExpression vrxp("^v\\s+" + num + "\\s+" + num + "\\s+" + num);
    QRegularExpression frxp("^f\\s+(\\d+)(?:/\\S*)?\\s+(\\d+)(?:/\\S*)?\\s+(\\d+)(?:/\\S*)?");

    while (!in.atEnd()) {
        QString line = in.readLine();
        auto vm = vrxp.match(line);
        if (vm.hasMatch()) {
            vertices.emplace_back(vm.captured(1).toDouble(),
                                  vm.captured(2).toDouble(),
                                  vm.captured(3).toDouble());
            continue;
        }
        auto fm = frxp.match(line);
        if (fm.hasMatch()) {
            // OBJ indices are 1-based; convert to 0-based.
            faces.emplace_back(fm.captured(1).toInt() - 1,
                               fm.captured(2).toInt() - 1,
                               fm.captured(3).toInt() - 1);
        }
    }
    file.close();
    return true;
}

MeshLoader::MeshLoader() {}
