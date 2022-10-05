import numpy as np

class Translator():
    '''
    Class to translate images in a given direction.
    The impact of the translation can also be set on instantiation.
    
    params:
        direction: direction in which the image will be translated.
        value: impact of the translation. higher values move the image further into given direction.
               IMPORTANT: values > 0.4 might not be label preserving.
    '''
    def __init__(self, direction:str='up', value:float=0.2) -> None:
        self.DIRECTIONS = ['up', 'down', 'left', 'right']

        self.set_direction(direction)
        self.set_value(value)


    def __repr__(self) -> str:
        return f'Translating {self.direction} by a value of {self.value}.'


    def set_direction(self, direction:str) -> None:
        if direction in self.DIRECTIONS:
            self.direction = direction
            self.__set_direction(direction)
        else:
            self.direction = 'up'
            self.__set_direction('up')
            print('Invalid direction entered.') 
            print(f'Direction set to default: {self.direction}')


    def __set_direction(self, direction:str) -> None:
        if direction in ['up', 'down']:
            self.__direction = self.__translate_vertical
        elif direction in ['left', 'right']:
            self.__direction = self.__translate_horizontal


    def set_value(self, value:str) -> None:
        if 0 <= value <= 1:
            self.value = value
            if value >= 0.4:
                print('NOTE: values >= 0.4 might not be label preserving!')
        else:
            self.value = 0.2
            print(f'Invalid value entered. Value must be between 0 and 1.')
            print(f'Value set to default: {self.value}')


    def __translate_vertical(self, img:np.ndarray) -> np.ndarray:
        '''
        Performs a vertical translation to an image in np.ndarray type.
        Direction can be either up or down.
        '''
        ud = int(img.shape[0] * self.value)
        if self.direction == 'up':
            img[:img.shape[0]-ud, :, :] = img[ud:, :, :]
            img[img.shape[0]-ud:, :, :] = 255
        elif self.direction == 'down':
            img[ud:, :, :] = img[:img.shape[0]-ud, :, :]
            img[:ud, :, :] = 255

        return img


    def __translate_horizontal(self, img:np.ndarray) -> np.ndarray:
        '''
        Performs a horizontal translation to an image in np.ndarray type.
        Direction can be either left or right.
        '''
        lr = int(img.shape[1] * self.value)
        if self.direction == 'left':
            img[:, :img.shape[1]-lr, :] = img[:, lr:, :]
            img[:, img.shape[1]-lr:, :] = 255
        elif self.direction == 'right':
            img[:, lr:, :] = img[:, :img.shape[1]-lr, :]
            img[:, :lr, :] = 255

        return img


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a translated image with given direction and percentage in np.ndarray type.
        '''
        return self.__direction(img)

