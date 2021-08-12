import shutil
import os
import re
import cv2
# will use them for creating custom directory iterator
import numpy as np
import pandas as pd
from six.moves import range
# regular expression for splitting by whitespace
splitter = re.compile("\s+")
base_path = '/DATA/Deepfashion_unzip_attr1000/img'
import cv2
import numpy as np
from PIL import Image
import random
import PIL.Image
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
import threading
from tensorflow.keras.utils import  Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open('/DATA/list_eval_partition.txt', 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]

    # Put each image into the relevant folder in train/test/validation folder
    for element in list_all:
      if not os.path.exists(os.path.join(base_path, element[2])):
        os.mkdir(os.path.join(base_path, element[2]))
      if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
        os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
      if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])), element[0].split('/')[0])):
        os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])), element[0].split('/')[0]))
      shutil.move(os.path.join(base_path, element[0]), os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0])) 
process_folders()
print ('OK')