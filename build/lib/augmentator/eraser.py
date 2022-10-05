from math import sqrt
import numpy as np
import re
import random

class Eraser():
    '''
    Erases one or more rectangles from random positions of an image.
    Fills erased areas according to a specified filling method.
    
    params:
        size: Size of areas to be erased as proportion of the whole image.
        count: Count of areas to be erased.
        fill: Method to use for filling of erased areas. Possible values:
            color: Use a specified color to fill the areas.
            gauss: Use a gaussian blur to fill the areas.
        color: Color to fill with
    '''

    def __init__(self, size:float=0.01, count:int=1, fill:str='color', color:str='#575757') -> None:
        # TODO: find fitting max_value
        # Constant values
        self.MAX_VALUE = 0.4
        self.FILL_TYPES = ['color', 'gauss']

        # parameter values
        self.set_size(size)
        self.set_count(count)
        self.set_fill(fill)
        self.set_color(color)


    def __repr__(self) -> str:
        return f'Filling with {self.fill}\nColor: {self.color}\nErasing {self.count} time{"s" if self.count > 1 else ""} with size {self.size}'


    def set_count(self, count:int) -> None:
        if 0 < count * self.size <= self.MAX_VALUE:
            self.count = count
        else:
            self.count = round(self.MAX_VALUE / self.size)
            print(f'Given count value is negative or would lead to erasion of more than {self.MAX_VALUE * 10}% of the image')
            print(f'Count set to max possible discrete value: {self.count}')


    def set_size(self, size:float) -> None:
        if 0 < size < self.MAX_VALUE:
            self.size = size
        else:
            self.size = 0.01
            print(f'Size must be a value between 0 and {self.MAX_VALUE}')
            print('Size set to default value: 0.01')


    def set_fill(self, fill:str) -> None:
        if fill in self.FILL_TYPES:
            self.fill = fill
        else:
            self.fill = 'color'
            print(f'Unknown fill type {fill}')
            print('Set fill to default: "color"')


    def set_color(self, color:str) -> None:
        if re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
            self.color = color
            self.color = self.__hex_to_rgb()
        else:
            self.color = '#575757'
            self.color = self.__hex_to_rgb()
            print('Invalid color String.')
            print(f'Set color to default: {self.color}')


    def __hex_to_rgb(self) -> tuple:
        '''
        Converts hex-variable 'color' to (R,G,B) tuple.
        '''
        return tuple(int(self.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


    def __color(self, img:np.ndarray) -> np.ndarray:
        '''
        Selects random positions for renctangles that are supposed to be erased.
        Fills those rectangles with a specified color
        params:
            img: Image to modify
        '''
        # erase position(s)
        img_shape = np.shape(img)
        max_y, max_x = img_shape[0], img_shape[1]
        rel_size = int(max_x * max_y * self.size)

        sqrt_rel_size = sqrt(rel_size)

        BUFFER = int(0.7 * sqrt_rel_size)

        for _ in range(self.count):
            x_len = random.randint(int(sqrt_rel_size - BUFFER), int(sqrt_rel_size + BUFFER))

            y_len = int(rel_size / x_len)

            start_x = random.randint(0, max_x- BUFFER)
            start_y = random.randint(0, max_y- BUFFER)
            end_x = int(min(max_x, start_x + x_len))
            end_y = int(min(max_y, start_y + y_len))

            for c in range(img_shape[2]):
                img[start_y:end_y, start_x:end_x, c] = self.color[c]
        return img


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Contoller for augmentation of the given image.
        params:
            img: Image to be augmented
        '''

        if self.fill == 'color':
            img = self.__color(img)

        elif self.fill == 'gauss':
            print('Gaussian filling not supported yet...')

        return img