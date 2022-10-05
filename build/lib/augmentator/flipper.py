import numpy as np

class Flipper():
    '''
    Class to flip images on a given axis.
    Flip can be performed on x, y or both axis.

    params:
        axis: axis to flip image on.
    '''
    def __init__(self, axis:str='y') -> None:
        axis = axis.lower()
        if axis in ['x', 'y', 'xy', 'yx']:
            self.set_axis(axis)
        else:
            self.axis = 'y'
            print('Invalid axis given, set default: y')

    def __repr__(self) -> str:
        return f'Flipping on {self.axis}-axis'
    
    def set_axis(self, axis:str) -> None:
        axis = axis.lower()
        if axis in 'xyx':
            self.axis = axis
        else:
            self.axis = 'y'
            print('Axis must be either x, y or both (xy, yx)')
            print(f'Axis set to default: {self.axis}')
    
    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a flipped image in np.ndarray type.
        '''
        if 'x' in self.axis:
            img = np.flip(img, axis=0)
        if 'y' in self.axis:
            img = np.flip(img, axis=1)
        return img
