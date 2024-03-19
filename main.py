from modbus import PLC, Camera, CAMERA_TRIGGER_REG, CAMERA_TRIGGER_OK, OK_RECALIBRATE_REG, enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
import  auto_roi.rois as ROIS
import cv2
import pickle
SHOW = True
mx, my = pickle.load(open('mp_xy.pkl', 'rb'))


class PROCESS:

    rois = {
        "200": loadRois("200_roi",(10,0)),
        "100": loadRois("100_roi",(0,0)),
    }

    @property
    def camera_read(camera):
        for _ in range(3):
            _info, _frame = camera.read()
            _frame = cv2.remap(_frame, mx, my, cv2.INTER_LINEAR)
        return _info, _frame
    
    @property
    def should_trigger():
        return PLC.read_coils(CAMERA_TRIGGER_REG, 1)[0]
    
    @property
    def should_recalibrate():
        return PLC.read_coils(OK_RECALIBRATE_REG, 1)[0]

    @property
    def get_model():
        return PLC.read_coils(68, 1)[0]
    
    @property
    def get_model_name():
        return '200' if PROCESS.get_model else '100'

    @property
    def get_roi_tag():
        return 'merge' if PROCESS.get_model else 'mask'

    @property
    def get_roi(self,):
        return PROCESS.rois[PROCESS.get_model_name].copy()

    def validade():
        _info, _frame = PROCESS.camera_read()
        if _info:
            return validate(_frame,  PROCESS.get_roi(), _default_rois, (10,10), 20)

    def update_roi():
        _info, _frame = PROCESS.camera_read()
        if _info:
            r, i = ROIS.new(img=_frame) 
            return ROIS.merge_with_old(PROCESS.get_roi(), r[PROCESS.get_roi_tag]), i

def process(camera):
    if PROCESS.should_trigger:
        return PROCESS.validade(camera)
    
    if PROCESS.should_recalibrate:
        r, i = PROCESS.update_roi()
        PROCESS.rois[PROCESS.get_model_name] = r

        with open(f"{PROCESS.get_model_name}_roi.new", 'wb') as file:
            pickle.dump(r, file)

    PLC.write_multiple_coils(CAMERA_TRIGGER_REG,[False,False])
    return None, None, None


if __name__ == '__main__':
    _default_steps = _default_steps()

    camera = Camera(False)
    with camera as camera:
        while True :
            try:
                _info, _new, old = process(camera)
                if _info:

                    presence =    [ frame.presence for frame in _info ][::-1]
                    orientation = [ frame.orientation for frame in _info ][::-1]
                    print(orientation)
                    enviar_valores_para_clp(100, presence, 20)
                    enviar_valores_para_clp(300, orientation, 20)
                    PLC.write_multiple_coils(CAMERA_TRIGGER_OK, [True])
                    cv2.imwrite('atual.jpg', old)
                    cv2.imwrite('process.jpg', _new)

            except IndexError:
                print("Index Error: modbus connection fail?")
                pass

