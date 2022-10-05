# augmentator

## Authors
Julius Ellermann
Jan Philipp Zimmmer

## Description
Project for the **Big** Data Praktikum with the topic *Image Data Augmentation* of the University of Leipzig<br>
This project provides a library with several basic image data augmentation methods like *rotation*, *translation* or *cropping* as well as the deep learning method *neural style transfer*.<br>
All of the implemented methods were mainly inspired by the paper [A survey on Image Data Augmentation for Deep Learning](https://doi.org/10.1186/s40537-019-0197-0).

## Installation
Navigate to the project folder of the augmentator and run the following commands:<br><br>
`python -m build`<br><br>
`pip install .`
___
## methods provided by the library
* color transformation
* cropping
* erasing
* flipping
* kernel filter
* mixing of two images
* noise injecting
* rotation
* translating
* *neural style transfer*

___
## structure of the library
All of the provided methods follow the same structure and should be fairly simmilar to use.<br>
![universal_class_diagram](/assets/augmentation_class.jpg)<br>
All of the classes follow more or less this 'universal' class diagram. Some basic classes only have two parameters that need to be set. A default is provided in every class. Some of the more complex augmentation methods provide more parameters that can be set. All parameters can be set on and after instantiation with the provided setter methods. This allows for correct parameter setting in the restrictions of the method.<br>
All methods have the **augmentation(img:np.ndarray)** in common. Every class is able to perform an augmentation on an image as np.ndarray with the *np.shape (H, W, RGB)* and returning an altered image of the same type and structure. Some methods return images with the same dimensions, some of them alter them.<br>

___
## usage and pipeline
The usage of the augmentation methods is simple and universal.<br>
For this project a pipeline is implemented that loads a *.json* config which automatically instantiates the corresponding classes with the specified parameters. The *.json* file needs to be in the following format:
```json
[
    {
        "class": "Eraser",
        "params": [0.02, 6, "rgb", "color", "#EB0240"]
    },
    {
        "class": "Noise_Injector",
        "params": ["gauss", 0.4]
    }
]
```
The augmentations are automatically performed on a provided image dataset. For this a *.csv* must be provided which contains the structure, path and labels of the data. As the implemented augmentations in the library all provide a predictable output, the pipeline does not need information on the methods used. After the augmentation, a dataset with the augmentated image data is created to train a convolutional neural network ([CNN](https://de.wikipedia.org/wiki/Convolutional_Neural_Network)).

___
## examples
Examples can be taken in the following order: `base_image, color_transformer, cropper, eraser, flipper, kernel_filter, noise_injector, translator, rotator, styler`<br>
<img src="/assets/base.jpg" height="200" height="200"/>
<img src="/assets/color_transformer.jpg" height="200" height="200"/>
<img src="/assets/cropper.jpg" height="200" height="200"/>
<img src="/assets/eraser.jpg" height="200" height="200"/>
<img src="/assets/flipper.jpg" height="200" height="200"/>
<img src="/assets/kernel_filter.jpg" height="200" height="200"/>
<img src="/assets/noise_injector.jpg" height="200" height="200"/>
<img src="/assets/translator.jpg" height="200" height="200"/>
<img src="/assets/rotator.jpg" height="200" height="200"/>
<img src="/assets/styler.jpg" height="200" height="200"/>
