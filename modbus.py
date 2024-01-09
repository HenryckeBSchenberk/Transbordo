from pyModbusTCP.client import ModbusClient

from cv2 import imread

SLAVE_ADDRESS = '192.168.10.102'
PORT = 502
UNIT_ID = 1

CAMERA_TRIGGER_REG=2

PLC = ModbusClient(host=SLAVE_ADDRESS, port=PORT, unit_id=UNIT_ID, auto_open=True)
prefix = 'C:/Users/Henrycke/Documents/GitHub/Transbordo/'
_frame = imread('C:/Users/Henrycke/Documents/GitHub/Transbordo/docs/asset/frame.jpg')

class Camera:
    def read():
        return _frame

