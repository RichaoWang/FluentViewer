import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import FluentUI

FluPopup {
    property string m_source: ""
    id:root
    width: parent.width/1.5
    height: parent.height/1.2
    padding: 10

    ColumnLayout{
        id:lay
        anchors.fill: parent

        FluArea{
            anchors.bottom: btn.top
            anchors.bottomMargin: 10
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right

            FluImage{
                property int theta: 5
                anchors.left: parent.left
                anchors.leftMargin: theta
                anchors.right: parent.right
                anchors.rightMargin: theta
                anchors.top: parent.top
                anchors.topMargin: theta
                anchors.bottom: parent.bottom
                anchors.bottomMargin: theta
                source: root.m_source
                fillMode: Image.PreserveAspectFit
            }
        }


        FluFilledButton{
            id:btn
            anchors.horizontalCenter: parent.horizontalCenter
            text:"关闭"
            implicitHeight: 40
            implicitWidth: 100
            onClicked: {
                close()
            }
        }

    }
}
