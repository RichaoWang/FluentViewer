#ifndef HUMANINFER_H
#define HUMANINFER_H

#include <QObject>
#include <Python.h>

class HumanInfer : public QObject
{
    Q_OBJECT
public:
    explicit HumanInfer(QObject *parent = nullptr);

    ~HumanInfer();
//signals:

private:
    PyObject* pInstance=nullptr;

};

#endif // HUMANINFER_H
