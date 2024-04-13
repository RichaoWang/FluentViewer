# This Python file uses the following encoding: utf-8

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import torch
import subprocess
import logging


class HumanBodyInfer():
    def __init__(self):
        cache_path = "./cache"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        logging.basicConfig(level=logging.INFO,  # 设置日志级别
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='app.log',  # 将日志写入文件
                            filemode='w')
        self.logger = logging.getLogger()

    def search_file(self, directory_path, key):
        # 遍历目录，找出带有 '_rect.txt' 的文件
        file = [os.path.abspath(os.path.join(directory_path, filename))
                for filename in os.listdir(directory_path)
                if filename.endswith(key)]
        return file[0]

    def getCachePath(self) -> str:
        return os.path.dirname(os.path.realpath(__file__)) + "\cache"

    def checkCudaIsAvailable(self) -> bool:
        return torch.cuda.is_available()

    def getObjPath(self) -> str:
        return "file:///" + self.search_file(self.getCachePath(), "_mesh.obj")

    def getPoseImagePath(self) -> str:
        return "file:///" + self.search_file(self.getCachePath(), "_pose.png")

    def getMeshImagePath(self) -> str:
        return "file:///" + self.search_file(self.getCachePath(), "_mesh.png")

    def checkExists(self, image_path):
        if not os.path.exists(image_path):
            return False
        for file_name in os.listdir(self.getCachePath()):
            file_path = os.path.join(self.getCachePath(), file_name)
            # 判断是否为文件，如果是则删除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        _, file_extension = os.path.splitext(image_path)
        tmp_img = cv2.imread(image_path)
        tmp_img_path = self.getCachePath() + "/" + os.path.basename(image_path)
        cv2.imwrite(tmp_img_path, tmp_img)
        return True

    def getPoseRectTxt(self) -> bool:
        files = os.listdir(self.getCachePath())
        if (len(files) != 1):
            return False
        tmp_image_path = files[0]

        cmd = r".\Python38\python.exe LightweightHumanPoseEstimation/run.py --image-path {}".format(
            self.getCachePath() + "\\" + tmp_image_path)

        self.logger.info(cmd)
        try:
            return_code = subprocess.call(cmd, shell=True)
            if return_code != 0:
                self.logger.error('getPoseRectTxt: {}'.format(subprocess.CalledProcessError(return_code, cmd)))
                return False
            self.logger.info("getPoseRectTxt: successful")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"getPoseRectTxt_1: Command failed with return code {e.returncode}")
            self.logger.error(f"getPoseRectTxt_2: Error output:\n{e.output.decode('utf-8')}")
            return False

    def getMeshFile(self) -> bool:
        files = os.listdir(self.getCachePath())
        if (len(files) != 3):
            self.logger.error("getMeshFile: 未生成依赖的三个文件")
            return False

        cmd = r".\Python38\python.exe  Pifuhd/run.py --use_rect -i {} -o {}".format(self.getCachePath(),
                                                                                self.getCachePath())

        # cmd = r".\Python38\python.exe -m Pifuhd.apps.simple_test --use_rect -i {} -o {}".format(self.getCachePath(),
        #                                                                             self.getCachePath())
        self.logger.info(cmd)
        try:
            return_code = subprocess.call(cmd, shell=True)
            if return_code != 0:
                self.logger.error('getMeshFile: {}'.format(subprocess.CalledProcessError(return_code, cmd)))
                return False
            self.logger.info("getMeshFile: successful")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"getMeshFile_1: Command failed with return code {e.returncode}")
            self.logger.error(f"getMeshFile_2: Error output:\n{e.output.decode('utf-8')}")
            return False


if __name__ == '__main__':
    a = HumanBodyInfer()
    print(a.checkExists(r"C:\Users\12168\Desktop\test_human_images\tmp2.png"))
    print(a.getPoseRectTxt())
    print(a.getMeshFile())
    # print(a.getObjPath())
    # print(a.getMeshImagePath())
    # print(a.getPoseImagePath())
