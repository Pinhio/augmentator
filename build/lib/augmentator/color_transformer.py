import numpy as np

class Color_Transformer():
    '''
    Color_Transformer comes with several methods and options to alter the 
    color scheme of pictures in a desired way.
    It works with an RGB representation of colors.
    params:
        method: Specification of transformation method. Possible values:
                min: no pixel value will be lower than specified value
                max: no pixel value will be higher than specified value
                kill: sets all values of specified channels to 0
                keep: sets all values of not specified channels to 0
                inc: increases the pixels values by amount specified in value param. maximum is 255
                dec: decreases the pixels values by amount specified in value param. minimum is 0
        value: specifies intensity of augmentation
        channel: specifies scoped channels for augmentation. can be r, g, b and any combination of them.
        is_value_percentage: indicates wether value is to be used as absolute value or percentage value
    '''

    def __init__(self, method:str='keep', value:int=10, channel:str='rgb', is_value_percentage:bool=False) -> None:

        # constants
        self.METHODS = ['min', 'max', 'kill', 'keep', 'inc', 'dec']

        # variables
        self.set_method(method)
        self.set_value(value)
        self.set_channel(channel)
        self.set_is_value_percentage(is_value_percentage)


    def __repr__(self) -> str:
        return f'Transforming colors with method {self.method}\nChannel: {self.channel}\nValue {self.value} used as {"percentage" if self.is_value_percentage else "absolute"} value'


    def set_method(self, method:str) -> None:
        if method in self.METHODS:
            self.method = method
        else:
            self.method = 'keep'
            print('Input for method is unknown')
            print(f'Use default method: {self.method}')


    def set_value(self, value:str) -> None:
        if value >= 0:
            self.value = value
        else:
            self.value = 10
            print('Input for value must be greater than 0')
            print(f'Use default value: {self.value}')


    def set_channel(self, channel:str) -> None:
        channel = channel.lower()
        self.channel = channel
        if len(channel) > 3:
            self.channel = 'rgb'
            print('Input for channel is too long. Max. 3 chars (r,g,b) allowed.')
            print(f'Channels set to default: {self.channel}')
        for c in channel:
            if not c in 'rgb':
                self.channel = 'rgb'
                print('Input for channel contains invalid character')
                print(f'Channels set to default: {self.channel}')
                break
        if len(channel) > len(set([i for i in channel])):
            self.channel = 'rgb'
            print('Input for channel contains multiples')
            print(f'Channels set to default: {self.channel}')
    

    def set_is_value_percentage(self, is_value_percentage:bool) -> None:
        self.is_value_percentage = is_value_percentage


    def __get_channel_encoding(self) -> list:
        '''
        Resolves channel variable to numpy indices.
        '''
        numerical_channel = []
        for c in self.channel:
            if c == 'r':
                numerical_channel.append(0)
            elif c == 'g':
                numerical_channel.append(1)
            elif c == 'b':
                numerical_channel.append(2)
        return numerical_channel


    def __kill(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Set pixel values of specified channels to 0.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        for chan in numeric_channel:
                img[:, :, chan] = 0
        return img


    def __keep(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Set pixel values of not specified channels to 0.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        for chan in range(3):
            if not chan in numeric_channel:
                img[:, :, chan] = 0
        return img


    def __min(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Set pixel values of specified channels to a specified minimum value, if they are below.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        for chan in numeric_channel:
            img[img[:, :, chan] < self.value, chan] = self.value
        return img


    def __max(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Set pixel values of specified channels to a specified maximum value, if they are above.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        for chan in numeric_channel:
            img[img[:, :, chan] > self.value, chan] = self.value
        return img


    def __inc(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Increase pixel values of specified channels by given amount or percentage.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        if self.is_value_percentage:
            img = np.array(img, dtype=np.float64)
            percent_val = self.value / 100
            for chan in numeric_channel:
                bool_filter_fitting_values = img[:, :, chan] + img[:, :, chan] * percent_val <= 255
                bool_filter_gets_full = img[:, :, chan] + img[:, :, chan] * percent_val > 255
                img[bool_filter_fitting_values, chan] += img[bool_filter_fitting_values, chan] * percent_val
                img[bool_filter_gets_full, chan] = 255
            img = np.array(img, dtype=np.int64)
        else:
            for chan in numeric_channel:
                bool_filter_fitting_values = img[:, :, chan] <= 255 - self.value
                bool_filter_gets_full = img[:, :, chan] > 255 - self.value
                img[bool_filter_fitting_values, chan] += self.value
                img[bool_filter_gets_full, chan] = 255
        return img


    def __dec(self, img:np.ndarray, numeric_channel:list) -> np.ndarray:
        '''
        Decreases pixel values of specified channels by given amount or percentage.
        params:
            img: Image to modify
            numeric_channel: List of channel indexes to modify
        '''
        if self.is_value_percentage:
            img = np.array(img, dtype=np.float64)
            percent_val = self.value / 100
            for chan in numeric_channel:
                bool_filter_fitting_values = img[:, :, chan] - img[:, :, chan] * percent_val >= 0
                bool_filter_gets_empty = img[:, :, chan] - img[:, :, chan] * percent_val < 0
                img[bool_filter_fitting_values, chan] -= img[bool_filter_fitting_values, chan] * percent_val
                img[bool_filter_gets_empty, chan] = 0
            img = np.array(img, dtype=np.int64)
        else:
            for chan in numeric_channel:
                bool_filter_fitting_values = img[:, :, chan] >= 0 + self.value
                bool_filter_gets_empty = img[:, :, chan] < 0 + self.value
                img[bool_filter_fitting_values, chan] -= self.value
                img[bool_filter_gets_empty, chan] = 0
        return img


    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Contoller for augmentation of the given image.
        params:
            img: Image to be augmented
        '''

        numeric_channel = self.__get_channel_encoding()

        if self.method == 'kill':
          img = self.__kill(img, numeric_channel)  

        elif self.method == 'keep':
            img = self.__keep(img, numeric_channel)

        elif self.method == 'min':
            img = self.__min(img, numeric_channel)

        elif self.method == 'max':
            img = self.__max(img, numeric_channel)

        elif self.method == 'inc':
            img = self.__inc(img, numeric_channel)

        elif self.method == 'dec':
            img = self.__dec(img, numeric_channel)

        return img