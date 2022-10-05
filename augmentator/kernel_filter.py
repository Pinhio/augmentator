import numpy as np


class KernelFilter():
    '''
    Class to kernel filter any given colored image.
    The strength of the filter can be set on instantiation

    params:
        method: kernel with which image will be filtered
        value: intensity of chosen filter method
    '''
    def __init__(self, method:str='gaussian_high_pass', value:float=0.5) -> None:
        self.METHODS = ['gaussian_high_pass', 'gaussian_low_pass']
        self.set_method(method)
        self.value = value


    def __repr__(self) -> str:
        return f'Kernel filtering with {self.method} by a value of {self.value}.'


    def set_method(self, method:str) -> None:
        if method in self.METHODS:
            self.method = method
        else:
            self.method = 'gaussian_high_pass'
            print('Invalid method entered.') 
            print(f'Method set to default: {self.method}')


    def set_value(self, value:str) -> None:
        if 0 <= value <= 1:
            self.value = value
        else:
            self.value = 0.5
            print(f'Invalid value entered. Value must be between 0 and 1.')
            print(f'Value set to default: {self.value}')


    def __is_odd(self, num:int) -> bool:
        '''
        Checks if a number is odd.
        '''
        return bool(num % 2)


    def __distance(self, p1, p2) -> float:
        '''
        Calculates the euclidian distance between two points
        '''
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def __gaussian_low_pass(self, img_shape) -> np.ndarray:
        '''
        Calculate the gaussian low pass kernel image for a given image shape.
        '''
        base = np.zeros(img_shape)
        rows, cols = img_shape
        d_zero = self.value * 100

        center = (rows/2, cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = np.exp(((-self.__distance((y,x), center)**2) / (2*(d_zero**2))))

        return base


    def __gaussian_high_pass(self, img_shape) -> np.ndarray:
        '''
        Calculate the gaussian high pass kernel image for a given image shape.
        '''
        base = np.zeros(img_shape)
        rows, cols = img_shape
        d_zero = self.value * 100

        center = (rows/2, cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1 - np.exp(((-self.__distance((y,x), center)**2) / (2*(d_zero**2))))

        return base


    def __get_kernel(self, img_shape) -> np.ndarray:
        if self.method == 'gaussian_high_pass':
            return self.__gaussian_high_pass(img_shape)
        elif self.method == 'gaussian_low_pass':
            return self.__gaussian_low_pass(img_shape)



    def __filter_image(self, img:np.ndarray, kernel_img:np.ndarray) -> np.ndarray:
        '''
        Method to filter a colored image with a given kernel.
        Each channel of the image is tranformed into its frequency domain.
        There it is shifted, multiplied with kernel and shifted back.
        Lastly the frequency domain is retransformed into the color channel.
        All color channels are the kernel filtered image.
        '''
        filtered = []
        for color_channel in range(3):
            # extract color channel r,g,b
            channel = img[:, :, color_channel]

            # perform fft on image channel
            channel = np.fft.fft2(channel)

            # shift image, apply filter and shift back
            channel = np.fft.fftshift(channel)
            channel = channel * kernel_img
            channel = np.fft.ifftshift(channel)
            
            # perform inverse fft on image channel
            channel = np.fft.ifft2(channel)

            filtered.append(np.abs(channel))
        
        # create filtered image from seperate channels
        filtered = np.stack(filtered, axis=-1).astype(int)
        
        return filtered


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a kernel filtered image in np.ndarray type.
        '''
        # check if any dimension of image is odd and make even
        if self.__is_odd(img.shape[0]):
            img = np.delete(img, 0, 0)
        if self.__is_odd(img.shape[1]):
            img = np.delete(img, 0, 1)

        # get filter
        kernel_img = self.__get_kernel((img.shape[0], img.shape[1]))

        # filter image
        img = self.__filter_image(img, kernel_img)

        return img