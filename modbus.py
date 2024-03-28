from pyModbusTCP.client import ModbusClient
from pypylon import pylon
# from pyModbusTCP.constants import Endian
import numpy as np

SLAVE_ADDRESS = '192.168.2.5'
PORT = 502
UNIT_ID = 1

CAMERA_TRIGGER_REG=66
CAMERA_TRIGGER_OK=67
MODEL_REGISTER=68
OK_RECALIBRATE_REG=69

PLC = ModbusClient(host=SLAVE_ADDRESS, port=PORT, unit_id=UNIT_ID, auto_open=True)
prefix = 'C:/Users/Henrycke/Documents/GitHub/Transbordo/'

from threading import Thread

def enviar_valores_para_clp(start_address, array, size=20, reg=False):
    subarrays = [array[i:i+size] for i in range(0, len(array), size)]

    # Enviar cada subarray para os registradores correspondentes
    for i, subarray in enumerate(subarrays):
        address = start_address + i * size
        #print(address, subarray)
        if reg:
            d = [int(i*100) for i in subarray]
            # print(d)
            PLC.write_multiple_registers(address, d)
        else:
            PLC.write_multiple_coils(address, subarray)

class Camera:
    def __init__(self, live_feed=True, fake_path=None):
        if fake_path is not None:
            import os
            os.environ['PYLON_CAMEMU'] = "1"
        self.__camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.__converter = pylon.ImageFormatConverter()
        self.__camera.Open()
        self.__frame = None
        self.hasNewFrame = False
        self.live_feed = live_feed
        if fake_path is not None:
            self.setup_fake(fake_path)
        else:
            pylon.FeaturePersistence.Load("cameraSettings.pfs", self.__camera.GetNodeMap(), True)
            
        self.__setting_up()
        if self.live_feed:
            Thread(target=self.__trigger, daemon=True).start()

    def setup_fake(self, path):
        self.__camera.ImageFilename = path
        self.__camera.ImageFileMode = "On"
        self.__camera.TestImageSelector = "Off"
        self.__camera.Width = 1456
        self.__camera.Height = 1088

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.__camera.Close()

    def __setting_up(self):
        self.__converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.__converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        if self.live_feed:
            self.__camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    def read(self):
        if not self.live_feed:
            return self.__trigger(1)
        return self.hasNewFrame, self.__frame

    def __trigger(self, qtd=0):
        if qtd>0:
            self.__camera.StartGrabbingMax(qtd)
        while self.__camera.IsGrabbing():
            grabResult = self.__camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            self.hasNewFrame = grabResult.GrabSucceeded()
            if grabResult.GrabSucceeded():
                self.__frame = self.__converter.Convert(grabResult).GetArray()
            grabResult.Release()

        return self.hasNewFrame, self.__frame
    
if __name__ == '__main__':
    # cam = pylon.InstantCamera()
    import os
    import cv2
    os.environ["PYLON_CAMEMU"] = "1"
    cam = Camera(True, "/home/jetson/Pictures/m3/A")
    while cv2.waitKey(1) != 27:
        _frame = cam.read()
        if _frame is not None:
            cv2.imshow('frame', _frame)
