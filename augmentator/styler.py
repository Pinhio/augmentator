import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image


class Styler():
    '''
    Class to style images with the neural style transfer technique.
    For this to work a model and at least one style must be available on disk.
    Once instanciated the objects augment method will take any image as np.ndarray,
    style it depending on PARAMS and returns a styled image in np.ndarray format.
    
    params:
        model_folder: location of model on disk.
        style_folder: location of style images on disk.
        style: style to use stored in style_folder.
    '''
    def __init__(self, style_folder:str, style:str, model_folder:str='model') -> None:
        # load model
        self.set_model(model_folder)

        # set styles location
        self.set_style_folder(style_folder)

        # set style to be used
        self.set_style(style)


    def __repr__(self) -> str:
        return f'Styling with the following style: {self.style}'


    def set_model(self, model_folder:str) -> None:
        if not os.path.isdir(model_folder):
            raise ValueError(f'Model folder not found. Could not load model.')
        try:
            self.model = hub.load(model_folder)
        except:
            print('Error in model loading.')


    def set_style(self, style:str) -> None:
        style_file = f'{self.style_folder}/{style}.jpg'
        if not os.path.isfile(style_file):
            raise ValueError(f'Style {style} not found. Could not load style')
        try:
            img_to_load = np.array(Image.open(f'{self.style_folder}/{style}.jpg'))
            self.style = self.load_image(img_to_load)
        except:
            print(f'Error in style loading.')
            


    def set_style_folder(self, style_folder:str) -> None:
        if os.path.isdir(style_folder):
            self.style_folder = style_folder
        else:
            raise ValueError(f'Style folder not found. Could not load styles.')


    def crop_center(self, image):
        '''
        Returns a cropped square image.
        Image dimensions depend on shorter axis.
        '''
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
        return image


    def load_image(self, img, image_size=(256, 256), preserve_aspect_ratio=True):
        '''
        Loads and preprocesses image.
            1. convert image to float32 numpy array.
            2. normalize to [0, 1] range.
            3. convert to tensor.
            4. crop to square.
            5. resize image to PARAM: image_size
        '''
        img = img.astype('float32')
        img /= 255.0

        img = tf.convert_to_tensor(img, dtype=tf.float32)[tf.newaxis, ...]
        img = self.crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img
    

    def augment(self, img:np.ndarray) -> np.ndarray:
        '''
        Takes an image in np.ndarray type as input.
        Returns a styled image with given model and style in np.ndarray type.
        '''
        content_img = self.load_image(img)
        model_output = self.model(tf.constant(content_img), tf.constant(self.style))
        stylized_img = np.squeeze(model_output[0])
        return stylized_img
