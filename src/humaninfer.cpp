#include "humaninfer.h"

HumanInfer::HumanInfer(QObject *parent)
    : QObject{parent}
{
    Py_SetPythonHome(L"D:\\anaconda3\\envs\\pifuhd_env"); // todo build

    Py_Initialize();
    if( !Py_IsInitialized() ){
        qDebug("python env init fail.");
        // todo

        return;
    }

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");//HumanInferCore.py模块的指针
    PyObject* pModule = PyImport_ImportModule("HumanInferCore");
    if (! pModule){
        PyErr_Print();
        return;
    }
    PyObject* pDict = PyModule_GetDict(pModule);
    if(!pDict) {
        PyErr_Print();
        return;
    }
    PyObject* pClass = PyDict_GetItemString(pDict, "HumanBodyInfer");
    if (!pClass) {
        qDebug("Cant find calc class./n");
        return;
    }
    //得到构造函数而不是类实例
    PyObject* pConstruct = PyInstanceMethod_New(pClass);
    if (!pConstruct) {
        qDebug("Cant find calc construct./n");
        return;
    }

    pInstance=PyObject_CallObject(pConstruct,NULL);

    qDebug("Human core init successful!");
    //    //    "s": 字符串参数
    //    //    "i": 整数参数
    //    //    "f": 浮点数参数
    //    //    "O": 任意对象参数
    //    //    PyObject_CallMethod(pInstance,"get_rect","s","C:\\Users\\12168\\Desktop\\qml\\FluentViewer\\Human3DGeneration\\temp\\BVNH6DTU00AJ0003.jpg","i",512);
    //    PyObject_CallMethod(pInstance,"run","s","C:\\Users\\12168\\Desktop\\test_human_images\\tmp1.jpg");
    //    PyObject_CallMethod(pInstance,"run","s","C:\\Users\\12168\\Desktop\\test_human_images\\tmp2.png");
    //    Py_Finalize();
}
