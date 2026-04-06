#pragma once

#include <QMainWindow>
#include "glwidget.h"

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    MainWindow(const QString &configPath);
    ~MainWindow();

private:

    GLWidget *glWidget;
};
