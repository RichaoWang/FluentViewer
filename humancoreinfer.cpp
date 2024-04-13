#include "humancoreinfer.h"
#include <QDebug>
#include <thread>
#include <QCoreApplication>
#include <iostream>


class PyThreadStateLock
{
public:
    PyThreadStateLock(void){
        state = PyGILState_Ensure();
    }
    ~PyThreadStateLock(void){
        PyGILState_Release(state);
    }
private:
    PyGILState_STATE state;

};

long long getTimeStamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return milliseconds.count();
}

std::string getPythonErrorString() {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    // Normalize the exception
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    // Convert the exception to a string
    PyObject *str_exc_value = PyObject_Str(pvalue);
    const char *exc_value_str = PyUnicode_AsUTF8(str_exc_value);

    // Release the acquired references
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    Py_XDECREF(str_exc_value);

    // Return the exception string as C++ std::string
    return std::string(exc_value_str);
}

HumanCoreInfer::HumanCoreInfer(QObject *parent)
        : QObject{parent} {
    QString appDirPath = QCoreApplication::applicationDirPath();
    auto pyenv = appDirPath + "/Python38";
    Py_SetPythonHome((wchar_t *) (pyenv.data()));
//    Py_SetPythonHome(L"D:\\anaconda3\\envs\\pifuhd_env");
    Py_Initialize();
    PyEval_InitThreads();//启用线程支持
    if (!Py_IsInitialized()) {
        emit errorMsg("初始化python环境异常");
        return;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");//HumanInferCore.py模块的指针
    pModule = PyImport_ImportModule("HumanInferCore");
    if (!pModule) {
        emit errorMsg("获取HumanInferCore.py脚本失败");
        return;
    }

    pDict = PyModule_GetDict(pModule);
    if (!pDict) {
        emit errorMsg("PyModule_GetDict失败");
        return;
    }
    pClass = PyDict_GetItemString(pDict, "HumanBodyInfer");
    if (!pClass) {
        emit errorMsg("HumanBodyInfer类找寻失败");
        return;
    }
    //得到构造函数而不是类实例
    pConstruct = PyInstanceMethod_New(pClass);
    if (!pConstruct) {
        emit errorMsg("获取HumanBodyInfer构造失败");
        return;
    }

    pInstance = PyObject_CallObject(pConstruct, NULL);
    if (!pInstance) {
        emit errorMsg("HumanBodyInfer实例化失败");
        return;
    }
}

HumanCoreInfer::~HumanCoreInfer() {
    if (future.isRunning()) {
        future.cancel();
        future.waitForFinished();
    }
    Py_CLEAR(pInstance);
    Py_CLEAR(pModule);
    Py_CLEAR(pDict);
    Py_CLEAR(pClass);
    Py_CLEAR(pConstruct);
    Py_Finalize();
}

void HumanCoreInfer::infer(QString image_path) {
    emit inferStart();
    QtConcurrent::run([=]() {
        Py_BEGIN_ALLOW_THREADS
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            PyObject *pObj = PyObject_CallMethod(pInstance, "checkExists", "s", image_path.toStdString().c_str());
            if (pObj != nullptr) {
                bool isTrue = (PyObject_IsTrue(pObj) != 0);
                if (isTrue) {
                    inferInfo("缓存目录生成完成");
                } else {
                    errorMsg("缓存目录生成失败,请检查缓存目录是否图片生成异常");
                }
            } else {
                PyErr_Print();
                errorMsg("checkExists 接口调用失败");
            }
            Py_DECREF(pObj);

            auto t1 = getTimeStamp();
            PyObject *pObj1 = PyObject_CallMethod(pInstance, "getPoseRectTxt", "");
            if (pObj1 != nullptr) {
                bool isTrue = (PyObject_IsTrue(pObj1) != 0);
                if (isTrue) {
                    auto t2 = getTimeStamp();
                    std::string millisecondsStr = std::to_string(t2 - t1);
                    inferInfo(QString("人体姿态估计生成完成,耗时: %1 ms").arg(millisecondsStr.data()));
                    emit poseEstimationSuccess();
                } else {
                    errorMsg("人体姿态估计生成失败");
                }
            } else {
                PyErr_Print();
                errorMsg("get_rect 接口调用失败");
            }
            Py_DECREF(pObj1);


            auto t3 = getTimeStamp();
            PyObject *pObj2 = PyObject_CallMethod(pInstance, "getMeshFile", "");
            if (pObj2 != nullptr) {
                bool isTrue = (PyObject_IsTrue(pObj2) != 0);
                if (isTrue) {
                    auto t4 = getTimeStamp();
                    std::string millisecondsStr1 = std::to_string(t4 - t3);
                    inferInfo(QString("Mesh生成完成,耗时: %1 ms").arg(millisecondsStr1.data()));
                    emit meshEstimationSuccess();
                } else {
                    PyErr_Print();
                    std::string errorString = getPythonErrorString();
                    emit errorMsg(errorString.data());
                    errorMsg("Mesh生成失败");
                }
            } else {
                PyErr_Print();
                errorMsg("get_obj 接口调用失败");
            }
            Py_DECREF(pObj2);

            PyGILState_Release(gstate);
            emit inferEnd();
        Py_END_ALLOW_THREADS

    });

}

QString HumanCoreInfer::getObjPath() {
    PyObject *pObj = PyObject_CallMethod(pInstance, "getObjPath", "");
    char *info;
    PyArg_Parse(pObj, "s", &info);
    Py_DECREF(pObj);
    return QString(info);
}

QString HumanCoreInfer::getPoseImagePath() {
    class PyThreadStateLock PyThreadLock;//获取全局锁
    char *info1;
    PyObject *pObj1 = PyObject_CallMethod(pInstance, "getPoseImagePath", "");
    PyArg_Parse(pObj1, "s", &info1);
    Py_DECREF(pObj1);
    return QString(info1);
}

QString HumanCoreInfer::getMeshImagePath() {
    class PyThreadStateLock PyThreadLock;//获取全局锁
    char *info1;
    PyObject *pObj1 = PyObject_CallMethod(pInstance, "getMeshImagePath", "");
    PyArg_Parse(pObj1, "s", &info1);
    Py_DECREF(pObj1);
    return QString(info1);
}

bool HumanCoreInfer::checkCudaIsAvailable() {
    PyObject *pObj = PyObject_CallMethod(pInstance, "checkCudaIsAvailable", "");
    if (pObj != nullptr) {
        bool isTrue = (PyObject_IsTrue(pObj) != 0);
        if (isTrue) {
            return true;
        } else {
            return false;
        }
    } else {
        std::string errorString = getPythonErrorString();
        emit errorMsg(errorString.data());
        emit errorMsg("check_cuda_is_available 接口调用失败");
    }
    Py_DECREF(pObj);
}
