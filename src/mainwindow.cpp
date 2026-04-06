#include "mainwindow.h"
#include <QHBoxLayout>

MainWindow::MainWindow(const QString &configPath)
{
    glWidget = new GLWidget();
    glWidget->setMeshPath(configPath);

    QHBoxLayout *container = new QHBoxLayout;
    container->addWidget(glWidget);
    this->setLayout(container);
}

MainWindow::~MainWindow()
{
    delete glWidget;
}
