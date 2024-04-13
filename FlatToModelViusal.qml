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

// todolists
// 1.拖拽图片后url问题 - ok
// 2.提示信息
// 3.打包问题
// 4.计算时等待问题（开一个新线程专门用于计算）- ok
// 5.gpu占用过高问题 - not ok
// 6.opt选项布局布局中问题 - ok
// 7.关于程序体积过大的问题（用py2exe打包压缩的方式？）

Item{
    id:root
    property string use_tip: "1.请确保上传的图像资源人物呈正面且背景单一,否则可能生成异常;\n2.支持拖拽模型文件至窗口进行本地查看;\n3.支持拖拽图像资源至Image控件或点击选择资源按钮进行图像资源的上传;\n4.自动旋转模式下,鼠标无法拖动模型,关闭自动旋转,鼠标左键可进行模型的拖拽移动,鼠标右键可进行模型的旋转,鼠标滚轮可进行模型的缩放;\n5.点击左上角重置模型可恢复模型初始姿态."
    property bool calcing: false
    property string file_path: ""

    Component.onCompleted: {
        var i=human_infer.checkCudaIsAvailable()
        if (i){
            showSuccess("检测到CUDA可用",4000)
        }else{
            dialog1.open()
        }
    }

    FluContentDialog{
        id:dialog1
        title:"CUDA不可用"
        message:"程序检测到CUDA不可用,一般情况下出现该问题原因为PC未装载显卡(可能导致的问题为推理过慢&渲染卡顿),程序将会自动调整为CPU计算与渲染!"
        buttonFlags:FluContentDialogType.NegativeButton
        negativeText: "确定"
        onNegativeClicked: {
            dialog1.close()
        }
    }

    BusyIndicator{
        id:busyindicator
        anchors.centerIn: parent
        width: 100
        height: 100
        running: root.calcing
    }

    Connections{
        target: human_infer
        function onInferStart(){
            showInfo("人像立体化计算开始",2000)
            mesh.source=""
            root.calcing=true
        }

        function onInferEnd(){
            showInfo("人像立体化计算完成",4000)
            mesh.source=human_infer.getObjPath()
            root.calcing=false
        }

        function onErrorMsg(errormsg){
            showError(errormsg,0,"bug待解决...")
        }

        function onInferError(){
            showError("计算失败",0,"失败原因:...")
        }

        function onInferInfo(infomsg){
            showSuccess(infomsg,3000)
        }

        function onPoseEstimationSuccess(){
            var i =human_infer.getPoseImagePath()
            cache_model.append({name:i})
        }

        function onMeshEstimationSuccess(){
            var i =human_infer.getMeshImagePath()
            cache_model.append({name:i})
        }
    }

    HumanCoreInfer{
        id:human_infer
    }


    CacheVisualWindow{
        id:cache_wind
    }

    FluArea{
        id:cache_area
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        width: parent.width/7
        height:cache_view.contentHeight

        ListView{
            id:cache_view
            anchors.fill: parent
            clip: true
            focus: true
            model: cache_model
            delegate: cache_delagete
            spacing: 5
            reuseItems: false
        }

        ListModel{
            id:cache_model
        }

        Component {
            id:cache_delagete
            FluArea{
                focus: true
                width: parent.width
                height: parent.width
                border.color: Qt.rgba(0,0,0,0)
                Image{
                    id:cache_img
                    width: parent.width-5
                    height: parent.height-5
                    anchors.centerIn: parent
                    fillMode: Image.PreserveAspectFit
                    source: name
                }
                MouseArea{
                    anchors.fill: parent
                    onClicked: {
                        cache_wind.m_source=cache_img.source
                        cache_wind.open()
                    }
                }
            }
        }
    }


    Scene3D{
        id:scene_3d
        anchors.fill: parent
        z:-100
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
                linearSpeed: rot_tog.checked?0:5
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
                source: ""
                onStatusChanged:(status)=> {
                                    /*if (status===Mesh.Error){
                                        mesh.source=""
                                        showError("加载资源失败",0,"请检查资源的完整性")
                                    }else */
                                    //                                    if(status===Mesh.Loading){
                                    //                                        showInfo("资源载入中,请稍后",2000)
                                    //                                    }else if(status===Mesh.Ready){
                                    //                                        showSuccess("资源载入完成",2000)
                                    //                                    }
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
                components: [mesh, material,transform]
            }

        }

    }


    Timer{
        interval: 1000
        running: true
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
            anchors.horizontalCenter: opt_lay.horizontalCenter
            text:"自动旋转"
            enabled: !root.calcing
            checked: false
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
            id:color_layout
            anchors.horizontalCenter: opt_lay.horizontalCenter
            spacing: 10
            FluText{
                text:"模型上色:"
                Layout.alignment: Qt.AlignVCenter
            }
            FluColorPicker{
                id:color_picker
                enabled: !root.calcing
                enableAlphaChannel:false
                Component.onCompleted: {
                    setColor("#CD5C5C")
                }
            }
        }

        RowLayout{
            id:opt_lay
            Rectangle{
                id:img_area
                width: btns.height
                height: btns.height
                border.color: "grey"
                border.width: 2
                radius: 2
                FluImage{
                    id:img
                    anchors.centerIn: parent
                    width: parent.width-4
                    height: parent.height-4
                    source: ""
                    fillMode: Image.PreserveAspectFit
                }

                DropArea{
                    id:img_drop_area
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
                            if (fileExtension !== "jpg"&fileExtension !== "png") {
                                event.accepted = false
                                return
                            }
                            return true
                        }
                    onDropped:
                        (event)=>{
                            var url = getUrlByEvent(event)
                            if(url !== ""){
                                img.source=url
                                root.file_path=url.toString().replace("file:///", "")
                            }
                        }

                }

            }


            ColumnLayout{
                id:btns
                spacing: 10
                FluButton{
                    id:local_res_btn
                    enabled: !root.calcing
                    text:"选择资源"
                    onClicked: {
                        file_dialog.open()
                    }
                }

                FluButton{
                    text:"开始转换"
                    enabled: !root.calcing
                    onClicked: {
                        if (root.file_path.toString()===""){
                            showWarning("未输入人像图片数据")
                            return
                        }
                        human_infer.infer(root.file_path)
                        root.file_path=""
                        cache_model.clear()  // mclear
                        cache_model.append({name:img.source.toString()})
                        img.source=""
                    }
                }

                FluButton{
                    text:"保存obj"
                    enabled: !root.calcing
                    onClicked: {
                        // todo
                        showSuccess("保存到 .... 完成",2000)
                    }
                }

            }
        }
    }

    FileDialog {
        id: file_dialog
        nameFilters: ["Obj files (*.jpg;*.png)"]
        folder: StandardPaths.writableLocation(StandardPaths.DocumentsLocation)
        onAccepted: {
            var fileUrl = file_dialog.currentFile
            img.source=fileUrl
            root.file_path=fileUrl.toString().replace("file:///", "")
        }
    }

    ExampleWindow{id:example_wind}

    FluContentDialog{
        id:dialog
        title:"使用说明"
        message:root.use_tip
        buttonFlags:FluContentDialogType.NegativeButton| FluContentDialogType.PositiveButton
        negativeText: "查看上传图像样例"
        onNegativeClicked: {
            showInfo("打开查看图像样例")
            example_wind.open()
        }

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

    FluIconButton{
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 10
        anchors.topMargin: 10
        iconSource:FluentIcons.SIPMove
        enabled: rot_tog.checked?false:true
        iconSize: 25
        text:"重置模型姿态"
        onClicked: {
            objReset()
        }
    }

    property real obj_scale: 1.0
    function objAdaptiveScreen(){
        //        transform.scale=1.0
        var i=mesh.geometry.maxExtent.y
        var i2=mesh.geometry.minExtent.y
        var k=1.4/(Math.abs(i)+Math.abs(i2))
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

