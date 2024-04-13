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

/**
  自适应屏幕功能 可能存在bug... 以后再说...
  **/


FluArea{
    id:root
    property string use_tip: "1.当前仅支持加载的模型类型为(*.obj;*.stl;*.ply);\n2.自动旋转状态下将会禁止进行鼠标自由拖拽的方式查看模型;\n3.关闭自动旋转后,可以鼠标右键拖拽的方式实现模型360°的自由旋转;\n4.关闭自动旋转后,点击模型可直接回退至模型初始位姿;\n5.支持导入网络模型文件;\n6.支持热加载模型文件,可直接拖拽模型文件至程序."

    BusyIndicator{
        id:busyindicator
        anchors.centerIn: parent
        width: 100
        height: 100
        running: mesh.status===Mesh.Loading?true:false
    }


    Scene3D{
        id:scene_3d
        anchors.fill: parent
        focus: true
        aspects: ["input", "logic"]
        cameraAspectRatioMode: Scene3D.AutomaticAspectRatio
        hoverEnabled: true
        Entity {
            Camera {
                id: camera
                projectionType: CameraLens.PerspectiveProjection
                fieldOfView: 22.5
                aspectRatio: scene_3d.width / scene_3d.height
                nearPlane: 1
                farPlane: 1000.0
                viewCenter: Qt.vector3d( 0.0, 0.0, 0.0 )
                upVector: Qt.vector3d( 0.0, 1.0, 0.0 )
                position: Qt.vector3d( 0.0, 0.0, 5.0 )
            }


            OrbitCameraController {
                id:contrl
                linearSpeed: 0
                lookSpeed: rot_tog.checked?0:1500
                zoomInLimit: 0
                camera: camera
            }

            components: [
                RenderSettings{
                    activeFrameGraph: ForwardRenderer{
                        id:renderer
                        clearColor: Qt.rgba(0,0,0,0);
                        camera: camera
                    }
                },
                InputSettings{}
            ]

            Mesh {
                id: mesh
                source: "https://zhu-zichu.gitee.io/test.obj"
                onStatusChanged:(status)=> {
                                    if (status===Mesh.Error){
                                        mesh.source=""
                                        showError("加载资源失败",0,"请检查资源的完整性")
                                    }else if(status===Mesh.None){
                                        showError("加载资源失败",0,"请检查资源的完整性")
                                    }else if(status===Mesh.Loading){
                                        showInfo("资源载入中,请稍后",2000)
                                    }else if(status===Mesh.Ready){
                                        showSuccess("资源载入完成",2000)
                                    }
                                }
            }

            PhongMaterial {
                id: material
                ambient: color_picker.colorValue
            }

            Transform{
                id:transform
                scale: 1.0
                translation: Qt.vector3d(0, 0, 0)
                rotation: fromEulerAngles(0, 0, 0)
                property real hAngle:0.0
                NumberAnimation on hAngle{
                    id:ani
                    from:0.0
                    to:360.0
                    duration: 5000
                    loops: Animation.Infinite
                    running: rot_tog.checked
                }
                matrix:{
                    var m=Qt.matrix4x4();
                    m.rotate(hAngle,Qt.vector3d(0,1,0));
                    m.translate(Qt.vector3d(0,0,0));
                    return m;
                }
                onMatrixChanged: {
                    objAdaptiveScreen()
                }
            }

            Entity {
                id: entity
                components: [mesh, material,transform,objPicker]
            }

            ObjectPicker{
                id: objPicker
                onClicked:(obj_pick)=>{
                              if (obj_pick.button!==1|rot_tog.checked===true){
                                  return
                              }
                              objReset()
                          }
            }
        }

    }


    Timer{
        interval: 1000
        running: transform.matrixChanged()?false:true
        repeat: true
        onTriggered: {
            objAdaptiveScreen()
        }
    }

    ColumnLayout{
        id:par_layout
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20

        FluToggleSwitch{
            id:rot_tog
            text:"自动旋转"
            checked: true
            onClicked:{
                if(checked){
                    objReset()
                    showSuccess("开启自动旋转")
                }else{
                    showSuccess("关闭自动旋转")
                }
            }
        }

        RowLayout{
            spacing: 10
            FluText{
                text:"模型上色:"
                Layout.alignment: Qt.AlignVCenter
            }
            FluColorPicker{
                id:color_picker
                enableAlphaChannel:false
                Component.onCompleted: {
                    setColor("#DDA0DD")
                }
            }
        }


        FluButton{
            id:local_res_btn
            text:"选择本地模型资源"
            onClicked: {
                file_dialog.open()
            }
        }

        FluButton{
            text:"加载网络模型资源"
            onClicked: {
                net_res_win.open()
            }
        }
    }

    NetResouceWindow{id:net_res_win;onPubNetPath:(p)=> {
                                                     mesh.source = p
                                                 }}

    FileDialog {
        id: file_dialog
        nameFilters: ["Obj files (*.obj;*.stl;*.ply)"]
        folder: StandardPaths.writableLocation(StandardPaths.DocumentsLocation)
        onAccepted: {
            var fileUrl = file_dialog.currentFile
//            console.log("qweqweqwe")
            console.log(fileUrl)
            mesh.source = fileUrl
        }
    }

    FluContentDialog{
        id:dialog
        title:"使用说明"
        message:root.use_tip
        buttonFlags: FluContentDialogType.PositiveButton
        positiveText:"返回"
        onPositiveClicked:{
            showSuccess("返回完成")
        }
    }

    FluIconButton{
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.rightMargin: 10
        anchors.topMargin: 10
        iconSource:FluentIcons.Help
        iconSize: 25
        text:"使用说明:\n"+root.use_tip
        onClicked: {
            dialog.open()
        }
    }

    property real obj_scale: 1.0

    function objAdaptiveScreen(){
        transform.scale=1.0
        var i=mesh.geometry.maxExtent.y
        var i2=mesh.geometry.minExtent.y
        //        abs()
        var k=0.9/(Math.abs(i)+Math.abs(i2))
        if (obj_scale!==k){
            obj_scale=k
        }
        transform.scale=obj_scale
    }

    function objReset(){
        transform.rotation=transform.fromEulerAngles(0, 0, 0)
        camera.viewCenter= Qt.vector3d( 0.0, 0.0, 0.0 )
        camera.upVector= Qt.vector3d( 0.0, 1.0, 0.0 )
        camera.position= Qt.vector3d( 0.0, 0.0, 5.0 )
    }


    DropArea{
        id:drop_area
        anchors.fill: parent

        function getUrlByEvent(event){
            var url = ""
            if (event.urls.length === 0) {
                url = "file:///"+event.getDataAsString("text/plain")
            }else{
                url = event.urls[0].toString()
            }
            return url
        }

        onEntered:
            (event)=>{
                if(!event.hasUrls){
                    event.accepted = false
                    return
                }
                var url = getUrlByEvent(event)
                if(url === ""){
                    event.accepted = false
                    return
                }
                var fileExtension = url.substring(url.lastIndexOf(".") + 1)
                if (fileExtension !== "obj"&fileExtension !== "stl"&fileExtension !== "ply") {
                    event.accepted = false
                    return
                }
                return true
            }
        onDropped:
            (event)=>{
                var url = getUrlByEvent(event)
                if(url !== ""){
                    mesh.source=url
                }
            }

    }

}

