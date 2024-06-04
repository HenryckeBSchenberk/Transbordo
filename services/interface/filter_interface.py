from interface.utils.split_data import (
    loadRois,
    normalizeData,
    applyRois
)

from interface.utils.datatypes import frame
import cv2

class ImageFilter:
    def __init__(self):
        self.__image = None
        self.__roi = None
        self.__result = None
        self.has_unread_result = False
        self.__last_roi_value = None
        self.__images = None

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, value):
        self.__image = value
        self.__update_result()
    
    @property
    def images(self):
        return self.__images

    @images.setter
    def images(self, value):
        self.__images = value
    
    @property
    def roi(self):
        return self.__roi
    
    @roi.setter
    def roi(self, value):
        if self.__last_roi_value != value:
            self.__roi = loadRois(f"/app/{value}", expand=(0,0))
            self.__last_roi_value = value
        # self.__update_result()

    @property
    def result(self):
        self.has_unread_result = False
        return self.__result

    @result.setter
    def result(self, value):
        self.__result = value
        self.has_unread_result = True

    def __update_result(self):
        if (self.image is not None) and (self.roi is not None):
            self.images = list(applyRois(self.image, self.roi))
            self.result = normalizeData(self.images)

    def get_info(self):
        print(len(self.images), len(self.result), len(self.roi))
        return [frame(original, normalized, roi) for original, normalized, roi in  zip(self.images, self.result, self.roi)]