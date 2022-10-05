import numpy as np
import random

class Cropper():
    ''' 
    Class to crop images.
    Cropping can be centered or in a random position.
    NOTE: Size of images is alterred after cropping.

    params:
        method: method to crop the image with. 
            valid values are center or random.
        value: set intensity of cropping.
            IMPORTANT: values >= 0.5 might not be label preserving.
    '''
    def __init__(self, method:str='center', value:float=0.3) -> None:
        self.METHODS = ['center', 'random']

        self.set_method(method)
        self.set_value(value)


    def __repr__(self) -> str:
        return f'Cropping at {self.method} with a value of {self.value}.'


    def set_method(self, method:str) -> None:
        if method.lower() in self.METHODS:
            self.method = method
            self.__set_method(method)
        else:
            self.method = 'center'
            self.__set_method('center')
            print('Invalid method entered.') 
            print(f'Method set to default: {self.method}')


    def __set_method(self, method:str) -> None:
        if method == 'center':
            self.__method = self.__center_crop
        elif method == 'random':
            self.__method = self.__random_crop


    def set_value(self, value:float) -> None:
        if 0 <= value <= 1:
            self.value = value
            if value >= 0.5:
                print('NOTE: values >= 0.5 might not be label preserving!')
        else:
            self.value = 0.3
            print(f'Invalid value entered. Value must be between 0 and 1.')
            print(f'Value set to default: {self.value}')


    def __random_crop(self, img:np.ndarray) -> np.ndarray:
        '''
        Crop images randomly in np.ndarray type.
            1. find a random position.
            2. determine vertical position in image.
            3. determine horizontal position in image.
            4. crop on both axis with given value.
        '''
        # randomize the position of cropping
        r_ud = random.uniform(0.0, 1.0)
        r_lr = random.uniform(0.0, 1.0)

        # crop is moved vertically
        value_ud = self.value / (1 + r_ud)
        u = int(img.shape[0] * value_ud)
        d = int(img.shape[0] * (1.0 - (self.value - value_ud)))

        # crop is moved horizontally
        value_lr = self.value / (1 + r_lr)
        l = int(img.shape[1] * value_lr)
        r = int(img.shape[1] * (1.0 - (self.value - value_lr)))

        # crop value on both axis
        # this obviously does NOT scale linearly. 
        return img[u:d, l:r, :]


    def __center_crop(self, img:np.ndarray) -> np.ndarray:
        '''
        Crop images in center by a given value in np.ndarray type.
        '''
        # determine values for new image edges
        half_val = self.value / 2
        u = int(img.shape[0] * (half_val))
        d = int(img.shape[0] * (1 - half_val))
        l = int(img.shape[1] * (half_val))
        r = int(img.shape[1] * (1 - half_val))

        # crop value on both axis
        # this obviously does NOT scale linearly. 
        return img[u:d, l:r, :]


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a cropped image in np.ndarray type.
        '''
        return self.__method(img)
