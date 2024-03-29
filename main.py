from modbus import PLC, Camera, CAMERA_TRIGGER_REG, enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
from codetiming import Timer
import cv2

SHOW = True

# @Timer(name="Main validation process")
def process(camera):
    should_trigger = True or PLC.read_input_registers(CAMERA_TRIGGER_REG)
    if should_trigger:
        _frame = camera.read()
        return validate(_frame,  _default_steps.copy(), _default_rois, False)

if __name__ == '__main__':
    _default_steps = _default_steps()
    _default_rois = loadRois("200_roi")

    camera = Camera()
    with camera as camera:
        while True:
            # if PLC.read_holding_registers('Y0.11', 1)[0]:
            _info, _new = process()
            
            presence =    [ frame.presence for frame in _info ]
            orientation = [ frame.orientation for frame in _info ]
            angle = [ frame.features.angle for frame in _info ]
            center = [ frame.features.center for frame in _info ]
            print(angle)
            print(center)

            enviar_valores_para_clp(100, presence, 20)
            enviar_valores_para_clp(200, orientation, 20)