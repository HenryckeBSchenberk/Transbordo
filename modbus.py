from pyModbusTCP.client import ModbusClient
from pypylon import pylon
# from pyModbusTCP.constants import Endian
import numpy as np

SLAVE_ADDRESS = '192.168.10.102'
PORT = 502
UNIT_ID = 1

CAMERA_TRIGGER_REG=2

PLC = ModbusClient(host=SLAVE_ADDRESS, port=PORT, unit_id=UNIT_ID, auto_open=True)
prefix = 'C:/Users/Henrycke/Documents/GitHub/Transbordo/'

from threading import Thread

def enviar_valores_para_clp(start_address, array, size=20):
    subarrays = [array[i:i+size] for i in range(0, len(array), size)]

    # Enviar cada subarray para os registradores correspondentes
    for i, subarray in enumerate(subarrays):
        address = start_address + i * size
        print(address, subarray)
        PLC.write_registers(address, subarray)

class Camera:
    def __init__(self, live_feed=True):
        self.__camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.__converter = pylon.ImageFormatConverter()
        self.live_feed = live_feed
        self.__setting_up()
        if self.live_feed:
            Thread(target=self.trigger, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self):
        self.__camera.Close()

    def __setting_up(self):
        self.__converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.__converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        if self.live_feed:
            self.__camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    def read(self):
        return self.__frame

    def trigger(self, qtd=0):
        if qtd>0:
            self.__camera.StartGrabbingMax(qtd)
        while self.__camera.IsGrabbing():
            grabResult = self.__camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                self.__frame = self.__converter.Convert(grabResult).GetArray()
            grabResult.Release()

        return grabResult.GrabSucceeded(), self.__frame