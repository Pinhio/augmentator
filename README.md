# augmentator

## Authors
Julius Ellermann<br>
Jan Philipp Zimmer

This is the installable version of our augmentation library called *augmentator*.<br>
For the classification pipeline we used it for, please refer to:
https://github.com/3LL3RM4NN/image_data_augmentation

## Description
Project for the module **Big** Data Praktikum (2022) with the topic *Image Data Augmentation* of the University of Leipzig<br>
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
* noise injection
* rotation
* translation
* *neural style transfer*

___
## Structure of the library
All of the provided methods follow the same structure and are fairly simmilar to use.<br>
![universal_class_diagram](/assets/augmentation_class.jpg)<br>
All of the classes follow this 'universal' class diagram. Some basic classes only have two parameters that need to be set. A default for each parameter is provided within every class. Some of the more complex augmentation methods contain additional parameters, as well as more than two available methods. All parameters can be set during instantiation but also changed later on with the provided setter methods. This allows for correct parameter setting in the restrictions of the method.<br>
All methods have the **augmentation(img:np.ndarray)**-method in common. Every class is able to perform an augmentation on an 3D np.ndarray with the *np.shape (H, W, RGB)* and returs an altered image of the same type and structure. Some methods return images with the same dimensions, others might alter them.<br>

___
## Usage and pipeline
The usage of the augmentation methods is simple and universal.<br>
A pipeline can be implemented that loads a *.json* config which instantiates the corresponding classes with the specified parameters.
For an example of a pipeline, please refer to the abovementioned repository.
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

___
## Examples
Examples sre shown in the following order: `base_image, color_transformer, cropper, eraser, flipper, kernel_filter, noise_injector, translator, rotator, styler`<br>
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
