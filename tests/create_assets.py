from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = np.array(Image.open('tests/images/butterfly.jpg'))

plt.figure()
plt.axis('off')
plt.imshow(img)
plt.savefig('assets/base.jpg', bbox_inches='tight', transparent=True, pad_inches=0)


# TODO: config that actually changes something
from augmentator.color_transformer import Color_Transformer
ct = Color_Transformer('kill', channel='r')

img_augment = ct.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/color_transformer.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.cropper import Cropper
c = Cropper()

img_augment = c.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/cropper.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.eraser import Eraser

e = Eraser(count=5)

img_augment = e.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/eraser.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.flipper import Flipper

f = Flipper()

img_augment = f.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/flipper.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.kernel_filter import KernelFilter

kf = KernelFilter()

img_augment = kf.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/kernel_filter.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


# TODO: Mixer


from augmentator.noise_injector import Noise_Injector

ni = Noise_Injector('salt_pepper')

img_augment = ni.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/noise_injector.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.rotator import Rotator

r = Rotator(angle=87)

img_augment = r.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/rotator.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.styler import Styler
s = Styler('tests/model', 'tests/images/styles', 'abstract')

img_augment = s.augment(img.copy())

plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/styler.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()


from augmentator.translator import Translator

t = Translator()

img_augment = t.augment(img.copy())
plt.figure()
plt.axis('off')
plt.imshow(img_augment)
plt.savefig('assets/translator.jpg', bbox_inches='tight', transparent=True, pad_inches=0)
# plt.show()