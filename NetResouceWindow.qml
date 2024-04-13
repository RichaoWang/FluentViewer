import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import FluentUI

FluPopup {
    id:root
    width: 400
    height: 200

    signal pubNetPath(string path)

    ColumnLayout{
        anchors.centerIn:  parent
        spacing: 10

        FluText{
            text: "请输入网络模型资源地址(精确到单个文件)"
            font.pixelSize: 20
        }

        FluText{
            text: "例如:"
            font.pixelSize: 15
        }

        FluCopyableText{
            text:"https://zhu-zichu.gitee.io/test.obj"
        }

        FluMultilineTextBox{
            id:box
            anchors.right: parent.right
            anchors.left:parent.left
            anchors.rightMargin: 20
            anchors.leftMargin: 20
            placeholderText:"请输入资源地址"
        }


        RowLayout{
            spacing: 30
            anchors.horizontalCenter: parent.horizontalCenter
            FluFilledButton{
                text:"确定"
                implicitHeight: 40
                implicitWidth: 100
                onClicked: {
                    if (box.text===""){
                        showError("资源地址为空,请输入正确且有效的资源地址")
                        return
                    }
                    pubNetPath(box.text)
                    close()
                }
            }

            FluButton{
                text:"返回"
                implicitHeight: 40
                implicitWidth: 100
                onClicked: {
                    close()
                    showInfo("取消网络模型资源加载")
                }
            }

        }
    }
}
