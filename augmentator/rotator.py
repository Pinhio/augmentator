# TODO: tests
# TODO: documentation

import numpy as np

class Rotator():

    def __init__(self, method:str='center', angle:int=90, pivot_point:tuple=None, anti_aliasing:bool=True) -> None:
        # constants
        self.METHODS = ['center', 'point']

        # setter
        self.set_method(method)
        self.set_anti_aliasing(anti_aliasing)
        self.set_angle(angle)
        self.set_pivot_point(pivot_point)

    def set_method(self, method:str) -> None:
        if method in self.METHODS:
            self.method = method
        else:
            self.method = 'center'
            print('Input for method is unknown')
            print(f'Use default method: {self.method}')

    def set_anti_aliasing(self, anti_aliasing:bool) -> None:
        if anti_aliasing and self.method == 'center':
            self.anti_aliasing = anti_aliasing
        elif not anti_aliasing:
            self.anti_aliasing = anti_aliasing
        else:
            self.anti_aliasing = False
            print('Error while setting anti_aliasing')
            print('anti_aliasing can only be used with method "center"')
            print(f'anti_aliasing set to {self.anti_aliasing}')

    def set_angle(self, angle:int) -> None:
        if 0 <= angle <= 360:
            self.angle = angle
        else:
            default = 90
            self.angle = default
            print('Illegal input for angle. Must be from 0 to 360.')
            print(f'Use default value: {default}')
    
    def set_pivot_point(self, pivot_point:tuple) -> None:
        if self.method == 'point':
            if 0 <= pivot_point[0] <= 100 and 0 <= pivot_point[1] <= 100:
                self.pivot_point_x = pivot_point[0]
                self.pivot_point_y = pivot_point[1]
            else:
                self.method = 'center'
                print('Illegal input for pivot_point. Must be from 0 to 100 (relative).')
                print(f'Change method to: {self.method}')

    def __rotate_center_anti_aliasing(self, img:np.ndarray) -> np.ndarray:

        angle = np.radians(self.angle)
        cosine = np.cos(angle)
        sine = np.sin(angle)

        width = img.shape[1]
        height = img.shape[0]

        new_height = round(abs(height * cosine) + abs(width * sine)) + 1
        new_width = round(abs(width * cosine) + abs(height * sine)) + 1

        new_img = np.zeros((new_height, new_width, img.shape[2]))
        new_img[:, :, :] = 255

        original_centre_height   = round(((height + 1) / 2) - 1)
        original_centre_width    = round(((width + 1) / 2) - 1)

        new_centre_height= round(((new_height + 1) / 2) - 1)
        new_centre_width= round(((new_width + 1) / 2) - 1)

        for h in range(height):
            for w in range(width):
                y = height - 1 - h - original_centre_height
                x = width - 1 - w - original_centre_width

                new_x, new_y = self.__shear(angle, x, y)

                new_y = new_centre_height - new_y
                new_x = new_centre_width - new_x

                new_img[new_y, new_x, :] = img[h, w, :]

        return new_img.astype(np.uint8)

    def __rotate_aliasing(self, img:np.ndarray) -> np.ndarray:
        # create a rotation matrix

        angle = np.radians(self.angle)
        rotation_matrix = np.transpose(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))

        w, h = img.shape[1], img.shape[0]

        pivot_point_x, pivot_point_y = self.__get_pivot_coordinates(img, w, h)

        new_img = np.zeros(img.shape, dtype='u1')
        new_img[:, :, :] = 255

        for height in range(h):
            for width in range(w):
                xy_matrix = np.array([[width - pivot_point_x], [height - pivot_point_y]])

                rotate_matrix = np.dot(rotation_matrix, xy_matrix)

                new_x = pivot_point_x + int(rotate_matrix[0])
                new_y = pivot_point_y + int(rotate_matrix[1])

                if (0 <= new_x <= w - 1) and (0 <= new_y <= h - 1):
                    new_img[new_y, new_x] = img[height, width]
        
        return new_img

    def __get_pivot_coordinates(self, img:np.ndarray, width, height) -> tuple:
        if self.method == 'center':
            return (int(width / 2), int(height / 2))
        else:
            return (int(width * (self.pivot_point_x / 100)), int(height * (self.pivot_point_y / 100)))


    def __shear(self, angle, x, y):
        # shear 1
        tangent = np.tan(angle / 2)
        new_x = round(x - y * tangent)
        new_y = y

        # shear 2
        new_y = round(new_x * np.sin(angle) + new_y)

        # shear 3
        new_x = round(new_x - new_y * tangent)

        return new_x, new_y

    def augment(self, img:np.ndarray) -> np.ndarray:
        if self.anti_aliasing and self.method == 'center':
            return self.__rotate_center_anti_aliasing(img)
        elif not self.anti_aliasing:
            return self.__rotate_aliasing(img)
        else:
            raise(ValueError('Antialiasing can not be used with method "point"'))