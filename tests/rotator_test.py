import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.mixer import Mixer

class TestMixer(unittest.TestCase):
    pass

### RUN
unittest.main()