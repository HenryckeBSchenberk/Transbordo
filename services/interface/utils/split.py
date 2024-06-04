from split_data import *
import numpy as np
import cv2
import random

def show(dataset):
    for img in dataset:
        cv2.imshow('f', img)
        cv2.waitKey(0)


# Machine = "A"
def get_raw_dataset(Machine, roi_name, local):
    return loadFrames(f"{local}/{Machine}/{roi_name}/NOK"), loadFrames(f"{local}/{Machine}/{roi_name}/OK"), loadFrames(f"{local}/{Machine}/{roi_name}/EMPTY"), loadRois(f"{local}/{Machine}/{roi_name}/{roi_name}_roi")

def get_splited_frames(datset, rois):
    all_frames = []
    for frame in datset:
        all_frames.extend(applyRois(frame, rois))
    # return np.array([all_frames).reshape(1, -1)
    return all_frames

def reverse(dataset):
    images = []
    for image in dataset:
        images.append(cv2.rotate(image, cv2.ROTATE_180))
    
    return images

def prepare_data(raw_dataset, roi):

    splited_data = get_splited_frames(raw_dataset, roi)
    normalized_data = normalizeData(splited_data)
    if len(normalized_data)>1:
        return np.array(normalized_data).reshape(-1,128,128,3)

def combine_reverse(a,b):
    sintetic_data = reverse(b)
    if len(a) == 0:
        return sintetic_data
    if len(b) == 0:
        return a
    return np.concatenate((np.array(a), np.array(sintetic_data)))

def create_train_test_sets(normalized_dataset, randomize=False, train_size=0.8):
    if randomize:
        random.shuffle(normalized_dataset)
    
    dataset_size = len(normalized_dataset)

    train_data = normalized_dataset[:int(train_size*dataset_size)]
    test_data = normalized_dataset[int(train_size*dataset_size):]

    return train_data, test_data


A_raw_images_nok_100, A_raw_images_ok_100, A_raw_images_empty_100, A_roi_100  =  get_raw_dataset("A", "100", "/app/dataset/raw")
B_raw_images_nok_100, B_raw_images_ok_100, B_raw_images_empty_100, B_roi_100  =  get_raw_dataset("B", "100", "/app/dataset/raw")
A_raw_images_nok_200, A_raw_images_ok_200, A_raw_images_empty_200, A_roi_200  =  get_raw_dataset("A", "200", "/app/dataset/raw")
B_raw_images_nok_200, B_raw_images_ok_200, B_raw_images_empty_200, B_roi_200  =  get_raw_dataset("B", "200", "/app/dataset/raw")

try:
    nok_dataset = np.concatenate(
                list(
                    data for data in (
                        prepare_data(A_raw_images_nok_100, A_roi_100), prepare_data(B_raw_images_nok_100, B_roi_100),
                        prepare_data(A_raw_images_nok_200, A_roi_200), prepare_data(B_raw_images_nok_200, B_roi_200)
                    ) if data is not None
                )
            )
except ValueError:
    nok_dataset = np.array([])

try:
    ok_dataset = np.concatenate(
                list(
                    data for data in (
                        prepare_data(A_raw_images_ok_100, A_roi_100), prepare_data(B_raw_images_ok_100, B_roi_100),
                        prepare_data(A_raw_images_ok_200, A_roi_200), prepare_data(B_raw_images_ok_200, B_roi_200)
                    ) if data is not None
                )
            )
except ValueError:
    ok_dataset = np.array([])

try:
    empty_dataset = np.concatenate(
                list(
                    data for data in (
                        prepare_data(A_raw_images_empty_100, A_roi_100), prepare_data(B_raw_images_empty_100, B_roi_100),
                        prepare_data(A_raw_images_empty_200, A_roi_200), prepare_data(B_raw_images_empty_200, B_roi_200)
                    ) if data is not None
                )
            )
except ValueError:
    empty_dataset = np.array([])

all_nok = combine_reverse(nok_dataset, ok_dataset)
all_ok = combine_reverse(ok_dataset, nok_dataset)
all_empty = combine_reverse(empty_dataset, empty_dataset)
# print(len(all_nok), len(all_ok), len(all_empty))
train_nok, test_nok = create_train_test_sets(all_nok, True)
train_ok, test_ok = create_train_test_sets(all_ok, True)
train_empty, test_empty = create_train_test_sets(all_empty, True)

saveFrames(train_nok, "/app/dataset/cnn/train/nok" )
saveFrames(test_nok, "/app/dataset/cnn/test/nok" )
saveFrames(train_ok, "/app/dataset/cnn/train/ok" )
saveFrames(test_ok, "/app/dataset/cnn/test/ok" )
saveFrames(train_empty, "/app/dataset/cnn/train/empty" )
saveFrames(test_empty, "/app/dataset/cnn/test/empty" )