from modbus import PLC, Camera, CAMERA_TRIGGER_REG, enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
from codetiming import Timer
import cv2
import numpy as np
from split_data import AdvancedRois

SHOW = True

# @Timer(name="Main validation process")
def process(camera):
    should_trigger = True or PLC.read_input_registers(CAMERA_TRIGGER_REG)
    if should_trigger:
        _info, _frame = camera.read()
        if _info and _frame is not None:
            rois = AdvancedRois(_frame, w=60,h=80,_min=30, _max=45, show=False)
            return validate(_frame,  _default_steps.copy(), rois, 20)
        return None, None
    
if __name__ == '__main__':
    _default_steps = _default_steps()
    _default_rois = loadRois("200_roi")

    #camera = Camera()
    camera = Camera(True, "D:\Images\m3\B")
    with camera as camera:
        while cv2.waitKey(1) !=27 :
            # if PLC.read_holding_registers('Y0.11', 1)[0]:
            _info, _new = process(camera)
            if _info:
                presence =    [ frame.presence for frame in _info ]
                orientation = [ frame.orientation for frame in _info ]
                angle = [ frame.features[0].angle if frame.presence else 0.00 for frame in _info ]
                center = [ (np.array(frame.features[0].center) - (np.array(frame.roi[2::])/2)) if frame.presence else (0, 0) for frame in _info ]

                cv2.imshow('org', _new)

            #enviar_valores_para_clp(100, presence, 20)
            #enviar_valores_para_clp(200, orientation, 20)
"""
# 100_roi
{
    w:57,
    h:139,
    r:0.2,
    _min:50,
    _max:80
}

# 200_roi
{
    w:55,
    h:77,
    r:0.7,
    _min:35,
    _max:40,
}
"""                