cmake_minimum_required(VERSION 3.16)

project(FluentViewer VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLUENTUI_BUILD_EXAMPLES OFF)

## 指定虚拟环境路径
set(Python3_ROOT_DIR "C:\\Users\\12168\\Desktop\\qml\\build-FluentViewer-Desktop_Qt_6_6_0_MSVC2019_64bit-Release\\Python38")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)


find_package(Qt6 6.2 COMPONENTS Quick Concurrent REQUIRED)

#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

qt_add_executable(appFluentViewer
    main.cpp
    src/AppInfo.cpp
    src/SettingsHelper.cpp
    src/FpsItem.cpp
    src/FpsItem.h
)

qt_add_qml_module(appFluentViewer
    URI FluentViewer
    VERSION 1.0
    QML_FILES main.qml
    QML_FILES ModelViusal.qml
    QML_FILES FlatToModelViusal.qml
    QML_FILES NetResouceWindow.qml
    SOURCES humancoreinfer.h humancoreinfer.cpp
    QML_FILES ExampleWindow.qml
    RESOURCES tempale.png
    QML_FILES CacheVisualWindow.qml
)


# 将推理py文件拷贝到构建目录
#file(COPY HumanInferCore.py DESTINATION ${CMAKE_BINARY_DIR})
configure_file(HumanInferCore.py HumanInferCore.py COPYONLY)
file(COPY LightweightHumanPoseEstimation DESTINATION ${CMAKE_BINARY_DIR})
file(COPY Pifuhd DESTINATION ${CMAKE_BINARY_DIR})

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
set_target_properties(appFluentViewer PROPERTIES
#    MACOSX_BUNDLE_GUI_IDENTIFIER com.example.appFluentViewer
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include_directories(${Python3_INCLUDE_DIRS})

target_link_libraries(appFluentViewer
    PRIVATE
    Qt6::Quick
    Qt6::Concurrent
    fluentuiplugin
    FramelessHelper::Core
    FramelessHelper::Quick
    ${Python3_LIBRARIES}

)

include(GNUInstallDirs)
install(TARGETS appFluentViewer
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_subdirectory(FluentUI)
