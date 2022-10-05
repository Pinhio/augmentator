import numpy as np

class Mixer():
    '''
    Mixes the pixel values of two images.
    params:
        method: Method to use for mixing. Possible values:
            avg: Use average pixel values of both images.
            rel: Use weighted values.
        value: Weight to use for method rel. rel with value 0.5 equals avg.
    '''
    
    def __init__(self, method:str='avg', value:float=0.5, mix_img:np.ndarray=None) -> None:
        
        # constant values
        self.METHODS = ['avg', 'rel']
        self.FILL_VALUE = 255

        # set attributes
        self.set_method(method)
        self.set_value(value)
        self.set_mix_img(mix_img)


    def __repr__(self) -> str:
        return f'Mixing two images with method {self.method}\nUsed mixing ratio for first image: {self.value}'


    def set_method(self, method:str) -> None:
        if method in self.METHODS:
            self.method = method
        else:
            self.method = 'avg'
            print('Invalid method entered.') 
            print(f'Method set to default: {self.method}')


    def set_value(self, value:float) -> None:
        if self.method != 'avg' and 0 <= value <= 1:
            self.value = value
        else:
            self.value = 0.5
            print(f'Invalid value entered. Value must be between 0 and 1.')
            print(f'Value set to default: {self.value}')


    def set_mix_img(self, mix_img:np.ndarray) -> None:
        # verify type
        if type(mix_img) != np.ndarray:
            raise ValueError(f'Can not Initialize Mixer: No mixing Image.')
        else:
            self.mix_img = mix_img


    def __reshape(self, img_1:np.ndarray, img_2:np.ndarray):
        '''
        Assimilates the shapes of img_1 and img_2 such, that they both are
            (max(img_1[x], img_2[x]), max(img_1[y], img_2[y], 3)
        Added areas contain solely white pixels.
        params:
            img_1, img_2: Images to reshape.
        '''

        # get shapes
        shape_1 = img_1.shape
        shape_2 = img_2.shape

        # no difference, no computation needed
        if shape_1 == shape_2:
            return img_1, img_2

        # get differences between images and desired values
        x_diff, y_diff, x_desired, y_desired = 0, 0, 0, 0
        if shape_1[0] != shape_2[0]:
            y_diff = abs(shape_1[0] - shape_2[0])
            y_desired = max(shape_1[0], shape_2[0])
        if shape_1[1] != shape_2[1]:
            x_diff = abs(shape_1[1] - shape_2[1])
            x_desired = max(shape_1[1], shape_2[1])

        # resize
        if y_diff != 0:
            # need to add rows
            add_on_top = y_diff // 2
            add_on_bottom = y_diff - add_on_top

            # height of img_1
            if shape_1[0] < y_desired:
                top_array = np.full(shape=(add_on_top, shape_1[1], 3), fill_value=self.FILL_VALUE, dtype=np.int32)
                bottom_array = np.full(shape=(add_on_bottom, shape_1[1], 3), fill_value=self.FILL_VALUE, dtype=np.int32)

                img_1 = np.append(top_array, img_1, axis=0)
                img_1 = np.append(img_1, bottom_array, axis=0)

            # height of img_2
            if shape_2[0] < y_desired:
                top_array = np.full(shape=(add_on_top, shape_2[1], 3), fill_value=self.FILL_VALUE, dtype=np.int32)
                bottom_array = np.full(shape=(add_on_bottom, shape_2[1], 3), fill_value=self.FILL_VALUE, dtype=np.int32)

                img_2 = np.append(top_array, img_2, axis=0)
                img_2 = np.append(img_2, bottom_array, axis=0)
            
        # update shapes to correctly compute array widths
        shape_1 = img_1.shape
        shape_2 = img_2.shape

        if x_diff != 0:
            # need to add columns
            add_left = x_diff // 2
            add_right = x_diff - add_left

            # width of img_1
            if shape_1[1] < x_desired:
                left_array = np.full(shape=(shape_1[0], add_left, 3), fill_value=self.FILL_VALUE, dtype=np.int32)
                right_array = np.full(shape=(shape_1[0], add_right, 3), fill_value=self.FILL_VALUE, dtype=np.int32)

                img_1 = np.append(left_array, img_1, axis=1)
                img_1 = np.append(img_1, right_array, axis=1)

            # width of img_2
            if shape_2[1] < x_desired:
                left_array = np.full(shape=(shape_2[0], add_left, 3), fill_value=self.FILL_VALUE, dtype=np.int32)
                right_array = np.full(shape=(shape_2[0], add_right, 3), fill_value=self.FILL_VALUE, dtype=np.int32)

                img_2 = np.append(left_array, img_2, axis=1)
                img_2 = np.append(img_2, right_array, axis=1)

        return img_1, img_2


    def __avg(self, img_1:np.ndarray, img_2:np.ndarray) -> np.ndarray:
        '''
        Method for mixing two images by averaging their pixel values.
        params:
            img_1, img_2: images to be mixed
        '''
        return np.add(img_1, img_2) // 2


    def __rel(self, img_1:np.ndarray, img_2:np.ndarray) -> np.ndarray:
        '''
        Method for mixing two images with a specified weight.
        params:
            img_1: image wich pixels are weighted self.value times
            img_2: image wich pixels are weighted 1 - self.value times
        '''
        mix_img = np.add(np.multiply(img_1, self.value), np.multiply(img_2, 1-self.value))
        # important: cast to int
        return mix_img.astype(np.int32)


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Contoller for augmentation of the given image.
        params:
            img: Image to be augmented
        '''

        # reshape images if necessary
        if img.shape != self.mix_img.shape:
            img, self.mix_img = self.__reshape(img, self.mix_img)

        # mix images
        if self.method == 'avg':
            img = self.__avg(img, self.mix_img)

        elif self.method == 'rel':
            img = self.__rel(img, self.mix_img)
        
        return img