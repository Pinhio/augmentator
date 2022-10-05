import numpy as np

class Noise_Injector():
    '''
    Class to inject noise to images.
    Noise can be either gaussian or salt_and_pepper.

    params:
        method: specifiy type of noise to be injected.
            valid values are either gauss or salt_pepper.
        value: intensity of imputed noise.
    '''
    def __init__(self, method:str='gauss', value:float=0.5) -> None:
        self.METHODS = ['gauss', 'salt_pepper']

        self.set_method(method)
        self.set_value(value)


    def __repr__(self) -> str:
        return f'Injecting noise with {self.method} and a value of {self.value}.'


    def set_method(self, method:str) -> None:
        if method in self.METHODS:
            self.method = method
            self.__set_method(method)
        else:
            self.method = 'gauss'
            self.__set_method('gauss')
            print('Invalid method entered.') 
            print(f'Method set to default: {self.method}')


    def __set_method(self, method:str) -> None:
        if self.method == 'gauss':
            self.__method = self.__gauss_noise
        elif self.method == 'salt_pepper':
            self.__method = self.__salt_pepper_noise

    
    def set_value(self, value:float) -> None:
        if 0 <= value <= 1:
            self.value = value
        else:
            self.value = 0.5
            print(f'Invalid value entered. Value must be between 0 and 1.')
            print(f'Value set to default: {self.value}')


    def __gauss_noise(self, img:np.ndarray) -> np.ndarray:
        '''
        Impute gaussian noise in images in np.ndarray type.
            1. create gaussian noise with image dimensions.
            2. add noise to image.
            3. clip image values to [0, 255]
        '''
        # TODO: find good values for noise strength (mean and sigma)
        mean = 30 * self.value
        sigma = 20 * self.value
        gauss = np.random.normal(mean, sigma, img.shape)

        img = img + gauss

        img = np.clip(img, 0, 255)

        return img.astype(np.int32)


    def __salt_pepper_noise(self, img:np.ndarray) -> np.ndarray:
        '''
        Impute salt_and_pepper noise in images in np.ndarray type.
            1. determine amount of noise.
            2. determine random spots for white (salt) pixels and impute.
            3. determine random spots for black (pepper) pixels and impute.
        '''
        amount = self.value / 100

        # salt
        num_salt = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape[:2]]
        img[tuple(coords)] = 255

        # pepper
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape[:2]]
        img[tuple(coords)] = 0

        return img


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a noise_injected image with specified noise method in np.ndarray type.
        '''
        return self.__method(img)
