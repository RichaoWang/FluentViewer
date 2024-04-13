import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Window
import FluentUI
import Qt3D.Core
import Qt3D.Render
import Qt3D.Input
import Qt3D.Extras
import QtQuick.Scene3D
import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick3D
import QtQuick3D.Effects
import QtQuick3D.Helpers

FluWindow {
    property int area_diff: 10

    id:window
    width: 1000
    height: 750
    title: "人体图像三维转换器"
    showDark: true

    FluArea{
        anchors.left: parent.left
        anchors.leftMargin: window.area_diff
        anchors.right: parent.right
        anchors.rightMargin: window.area_diff
        anchors.top: parent.top
        anchors.topMargin: 5
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        paddings: 10
        radius: 20

        FlatToModelViusal{
            anchors.fill: parent
        }
    }


    FpsItem{
        id:fps_item
    }

    FluText{
        text:"帧率 %1".arg(fps_item.fps)
        opacity: 0.5
        anchors{
            bottom: parent.bottom
            right: parent.right
            bottomMargin: 3
            rightMargin: 10
        }
    }

    FluText{
        text:"版本号 v0.0.1"
        opacity: 0.5
        anchors{
            bottom: parent.bottom
            left: parent.left
            bottomMargin: 3
            leftMargin: 10
        }
    }



}
