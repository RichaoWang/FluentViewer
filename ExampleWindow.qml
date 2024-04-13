import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import FluentUI

FluPopup {
    id:root
    padding: 10
    ColumnLayout{
        anchors.fill: parent
        spacing: 10

        FluImage{
            width: 400
            height: 400
            source: "tempale.png"
            fillMode: Image.PreserveAspectFit
        }

        FluFilledButton{
            anchors.horizontalCenter: parent.horizontalCenter
            text:"返回"
            implicitHeight: 40
            implicitWidth: 100
            onClicked: {
                close()
            }
        }

    }
}
