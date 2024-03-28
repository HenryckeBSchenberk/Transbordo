import cv2
import auto_roi.circles as circles
import numpy as np
import auto_roi.utils as utils
import pickle

ALL_LOAD = False
# picture = 2
# model = "200"
# original = cv2.imread(f"./calibration/test/New Folder/{model}/{model} ({picture}).png")

rois = {

}

model_draw = {
    "100":{
        "draw":{"_r":0.2, "_w":55,"_h":139, "group_distance":40},
        "mask_roi":{"_min":53, "_max":80, "group_distance":24, "a_min":700, "a_max":7500},
        "auto_roi": None,
    },
    "200":{
        "draw":{"_r":0.7, "_w":60,"_h":80},
        "mask_roi":{"_min":32, "_max":45, "group_distance":28, "a_min":700, "a_max":7500},
        "auto_roi":{"minDist":20,"param1":40,"param2":19,"minRadius":27,"maxRadius":32},
        # "auto_roi":{"minDist":17,"param1":65,"param2":19,"minRadius":27,"maxRadius":33},
    }
}


def sort(matrix):
    row_indices = np.argsort(matrix[:, :, 1], axis=1)
    rows, cols = np.meshgrid(np.arange(10), np.arange(20))
    sorted_matrix_y = np.take_along_axis(matrix, row_indices[:, :, np.newaxis], axis=1)
    col_indices = np.argsort(sorted_matrix_y[:, :, 0], axis=1)
    sorted_matrix = np.take_along_axis(sorted_matrix_y, col_indices[:, :, np.newaxis], axis=1)
    return sorted_matrix

def update_rois():
    global rois
    if model_draw[model]["auto_roi"] is not None:
        auto_roi_c = circles.detect(img, **model_draw[model]["auto_roi"])
        _, auto_rois, _, _ = circles.draw(img_draw_a, auto_roi_c, **model_draw[model]["draw"], color=(0, 0, 255))
        rois["auto"] = auto_rois

def new(*args, _img=None, model="100"):
    global rois
    img_draw = original.copy() if _img is None else _img.copy()
    img = original.copy() if _img is None else _img.copy()

    strategy = 0
    if model_draw[model]["auto_roi"] is not None:
        auto_roi_c = circles.detect(img, **model_draw[model]["auto_roi"])
        _, auto_rois, _, _ = circles.draw(img_draw, auto_roi_c, **model_draw[model]["draw"], color=(0, 0, 255))
        rois["auto"] = auto_rois
        # cv2.imshow('MoughCircle', cv2.resize(img_draw, None, None, 0.8, 0.8))
        strategy+=1

    if model_draw[model]["mask_roi"] is not None:
        c,  mk = circles.detect_2(img,  **model_draw[model]["mask_roi"])
        _, mask_rois, _, _ = circles.draw(img_draw, np.array([c]), **model_draw[model]["draw"], p=2, color=(255,0,0))
        rois["mask"] = mask_rois
        strategy+=1
    
    if strategy == 2:
        x = np.concatenate((np.array([c]), np.array([auto_roi_c[0, :, :2]])),  axis=1)
        new_c, _, _ = circles.centers(x, 2, 40)
        new_c = utils.organizar_matriz(new_c[0, :200], 10, 20)
        _, merge_rois, _, _ = circles.draw(img_draw, new_c.reshape(1,-1,2), **model_draw[model]["draw"], group_distance=50, p=2, color=(0,0,0))
        rois["merge"] = merge_rois
    
    #cv2.imshow("rois", img_draw)
    return rois, img_draw

def nothing(*args):
    return

def get_values(*args):
    if ALL_LOAD:
        global model_draw
        if model_draw[model]["auto_roi"] is not None:
            model_draw[model]["auto_roi"] = {
                "minDist":cv2.getTrackbarPos('minDist', 'AutoRoi'),
                "param1":cv2.getTrackbarPos('param1', 'AutoRoi'),
                "param2":cv2.getTrackbarPos('param2', 'AutoRoi'),
                "minRadius":cv2.getTrackbarPos('minRadius', 'AutoRoi'),
                "maxRadius":cv2.getTrackbarPos('maxRadius', 'AutoRoi'),
            }
        if model_draw[model]["mask_roi"] is not None:
            model_draw[model]["mask_roi"] = {
                "_min":cv2.getTrackbarPos('_min', 'MaskRoi'),
                "_max":cv2.getTrackbarPos('_max', 'MaskRoi'),
                "group_distance":cv2.getTrackbarPos('group_distance', 'MaskRoi'),
                "a_min":cv2.getTrackbarPos('a_min', 'MaskRoi'),
                "a_max":cv2.getTrackbarPos('a_max', 'MaskRoi'),
            }
            new()

def merge_with_old(old_rois, new_roi):
    missing_cords_on_new = utils.pontos_faltando(np.array(old_rois), np.array(new_roi), 20)
    if len(missing_cords_on_new) > 0:
        return np.concatenate((new_roi, missing_cords_on_new))
    return new_roi

if __name__ == "__main__":
    import argparse
    
    cv2.namedWindow('frame')
    mx, my = pickle.load(open('../calibration/mp_xy.pkl', 'rb')) 

    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-P', '--path', type=str, help='Path for image raw image')
    parser.add_argument('-M', '--model', type=str, help='Model')
    args = parser.parse_args()
    

    model = args.model 
    original = cv2.imread(args.path)
    # cv2.imshow("f", original)
    # cv2.waitKey(0)
    original = cv2.remap(original, mx, my, cv2.INTER_LINEAR)

    confident = 'merge' if model == '200' else 'mask'
    with open(f"../{model}_roi", 'rb') as file:
        old_rois =  pickle.load(file)

    if model_draw[model]["auto_roi"] is not None:
        cv2.namedWindow("AutoRoi")
        cv2.resizeWindow("AutoRoi", 700, 300)
        cv2.createTrackbar('minDist', 'AutoRoi', model_draw[model]["auto_roi"]["minDist"], 255, get_values)
        cv2.createTrackbar('param1', 'AutoRoi', model_draw[model]["auto_roi"]["param1"], 255, get_values)
        cv2.createTrackbar('param2', 'AutoRoi', model_draw[model]["auto_roi"]["param2"], 255, get_values)
        cv2.createTrackbar('minRadius', 'AutoRoi', model_draw[model]["auto_roi"]["minRadius"], 255, get_values)
        cv2.createTrackbar('maxRadius', 'AutoRoi', model_draw[model]["auto_roi"]["maxRadius"], 255, get_values)


    if model_draw[model]["mask_roi"] is not None:
        cv2.namedWindow("MaskRoi")
        cv2.resizeWindow("MaskRoi", 700, 300)
        cv2.createTrackbar("_min", 'MaskRoi', model_draw[model]["mask_roi"]["_min"], 255, get_values)
        cv2.createTrackbar("_max", 'MaskRoi', model_draw[model]["mask_roi"]["_max"], 255, get_values)
        cv2.createTrackbar("group_distance",  'MaskRoi', model_draw[model]["mask_roi"]["group_distance"], 255, get_values)
        cv2.createTrackbar("a_min", 'MaskRoi', model_draw[model]["mask_roi"]["a_min"], 15000, get_values)
        cv2.createTrackbar("a_max", 'MaskRoi', model_draw[model]["mask_roi"]["a_max"], 15000, get_values)
    
    ALL_LOAD = True
    new()        
    while cv2.waitKey(1) != 27:
        merge_with_old(np.array(old_rois), np.array(rois[confident]))
        pass
    else:
        for key, roi in rois.items():
            with open(f"{model}_roi.{key}", 'wb') as file:
                pickle.dump(roi, file)
