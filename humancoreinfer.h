#ifndef HUMANCOREINFER_H
#define HUMANCOREINFER_H


#include <Python.h>
#include <QObject>
#include <QtConcurrent>
#include <chrono>
#include <ctime>


class HumanCoreInfer : public QObject
{
    Q_OBJECT
public:
    explicit HumanCoreInfer(QObject *parent = nullptr);

    ~HumanCoreInfer();

    Q_INVOKABLE void infer(QString image_path);
    Q_INVOKABLE QString getObjPath();
    Q_INVOKABLE QString getPoseImagePath();
    Q_INVOKABLE QString getMeshImagePath();
    Q_INVOKABLE bool checkCudaIsAvailable();

signals:
    void inferStart();
    void inferEnd();
    void inferError();
    void inferInfo(QString);
    void errorMsg(QString);
    void poseEstimationSuccess();
    void meshEstimationSuccess();

private:
    PyObject* pInstance;
    PyObject* pModule;
    PyObject* pDict;
    PyObject* pClass;
    PyObject* pConstruct;
    QFuture<void> future;
};

#endif // HUMANCOREINFER_H
