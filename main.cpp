#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <FramelessHelper/Quick/framelessquickmodule.h>
#include <FramelessHelper/Core/private/framelessconfig_p.h>
#include <QObject>
#include "src/SettingsHelper.h"
#include "src/FpsItem.h"
#include "humancoreinfer.h"

FRAMELESSHELPER_USE_NAMESPACE

    int main(int argc, char *argv[])
{
    //将样式设置为Basic，不然会导致组件显示异常
    qputenv("QT_QUICK_CONTROLS_STYLE","Basic");
    FramelessHelper::Quick::initialize();
    QGuiApplication::setOrganizationName("WangRiChaoRen");
    QGuiApplication::setOrganizationDomain("https://github.com/wangrichaoren");
    QGuiApplication::setApplicationName("FluentViewer");
    QGuiApplication app(argc, argv);
#ifdef Q_OS_WIN // 此设置仅在Windows下生效
    FramelessConfig::instance()->set(Global::Option::ForceHideWindowFrameBorder);
#endif
    FramelessConfig::instance()->set(Global::Option::DisableLazyInitializationForMicaMaterial);
    FramelessConfig::instance()->set(Global::Option::CenterWindowBeforeShow);
    FramelessConfig::instance()->set(Global::Option::ForceNonNativeBackgroundBlur);
    FramelessConfig::instance()->set(Global::Option::EnableBlurBehindWindow);
#ifdef Q_OS_MACOS
    FramelessConfig::instance()->set(Global::Option::ForceNonNativeBackgroundBlur,false);
#endif
    QQmlApplicationEngine engine;
    FramelessHelper::Quick::registerTypes(&engine);
    const QUrl url(u"qrc:/FluentViewer/main.qml"_qs);
    qmlRegisterType<FpsItem>("FluentViewer", 1, 0, "FpsItem");
    qmlRegisterType<HumanCoreInfer>("FluentViewer", 1, 0, "HumanCoreInfer");
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
