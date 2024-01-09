from modbus import PLC, Camera, CAMERA_TRIGGER_REG
from detect import validate, _default_rois, _default_steps
from codetiming import Timer
import cv2

SHOW = True

# @Timer(name="Main validation process")
def process():
    should_trigger = True or PLC.read_input_registers(CAMERA_TRIGGER_REG)
    if should_trigger:
        _frame = Camera.read()
        return validate(_frame,  _default_steps.copy(), _default_rois, 20)

if __name__ == '__main__':
    _default_steps = _default_steps()
    _default_rois = _default_rois()

    while True:
        _, _new = process()

        if SHOW:
            cv2.imshow('frame', _new)
            cv2.waitKey(1)