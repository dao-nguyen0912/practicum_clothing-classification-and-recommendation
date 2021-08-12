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
import cv2
import numpy as np
from PIL import Image
import random
import PIL.Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Softmax, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD

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
#from sklearn.model_selection import train_test_split

X_train_filename =pd.read_csv('DAO/df_3categories_120000_train.csv')
X_test_filename =pd.read_csv('DAO/df_3categories_120000_test.csv')
#X = df_cate2_concat.loc[:,['img_name', 'img_cate2_name','x1','y1','x2','y2']]
#X_train_filename, X_test_filename = train_test_split(X, test_size=0.2, random_state=1, stratify=X['img_cate2_name'])

y_train =X_train_filename.loc[:,['img_cate1_code']]
y_val =X_test_filename.loc[:,['img_cate1_code']]
y_train =pd.get_dummies(y_train)
y_val =pd.get_dummies(y_val)

Filter_dftrain=X_train_filename
Filter_dftrain=Filter_dftrain.set_index('img_name')
print(len(Filter_dftrain))

Filter_dfval =X_test_filename
Filter_dfval=Filter_dfval.set_index('img_name')
print(len(Filter_dfval))
os.mkdir('DATA/DeepFashion_crop_cate1/')
os.mkdir('DATA/DeepFashion_crop_cate1/img')
os.mkdir('DATA/DeepFashion_crop_cate1/img/train')
os.mkdir('DATA/DeepFashion_crop_cate1/img/train/FULL_BODY')
os.mkdir('DATA/DeepFashion_crop_cate1/img/train/UPPER_BODY')
os.mkdir('DATA/DeepFashion_crop_cate1/img/train/LOWER_BODY')
os.mkdir('DATA/DeepFashion_crop_cate1/img/val')
os.mkdir('DATA/DeepFashion_crop_cate1/img/val/FULL_BODY')
os.mkdir('DATA/DeepFashion_crop_cate1/img/val/UPPER_BODY')
os.mkdir('DATA/DeepFashion_crop_cate1/img/val/LOWER_BODY')
for i in range(0, len(Filter_dftrain)):
  if os.path.exists(os.path.join(os.path.join('DATA/Deepfashion_unzip_new/img', Filter_dftrain.index[i].split('/')[1]),Filter_dftrain.index[i].split('/')[-1])):
    f_image = Image.open('DATA/Deepfashion_unzip_new/img/'+Filter_dftrain.index[i].split('/')[1]+'/'+Filter_dftrain.index[i].split('/')[-1])
    left = Filter_dftrain['x1'][i]
    top = Filter_dftrain['y1'][i]
    right = Filter_dftrain['x2'][i]
    bottom = Filter_dftrain['y2'][i]
    f_image = f_image.crop((left, top, right, bottom))
    f_image = f_image.resize((200, 200))
#    list_img.append([Filter_dftrain.index[i],f_image])
    foldercheck = Filter_dftrain.index[i].split('/')[0]
    foldercheck1 = Filter_dftrain.index[i].split('/')[1]
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/FULL_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/FULL_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/UPPER_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/UPPER_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/LOWER_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/LOWER_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/FULL_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/FULL_BODY/' + foldercheck+'/'+ foldercheck1)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/UPPER_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/UPPER_BODY/' + foldercheck+'/'+ foldercheck1)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/train/LOWER_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/train/LOWER_BODY/' + foldercheck+'/'+ foldercheck1)
    f_image.save('DATA/DeepFashion_crop_cate1/img/train/'+Filter_dftrain.img_cate1_name[i]+'/'+Filter_dftrain.index[i])

print('save crop_train_images ok')      
for i in range(0, len(Filter_dfval)):
  if os.path.exists(os.path.join(os.path.join('DATA/Deepfashion_unzip_new/img', Filter_dfval.index[i].split('/')[1]),Filter_dfval.index[i].split('/')[-1])):
    f_image = Image.open('DATA/Deepfashion_unzip_new/img/'+Filter_dfval.index[i].split('/')[1] +'/'+Filter_dfval.index[i].split('/')[-1])
    left = Filter_dfval['x1'][i]
    top = Filter_dfval['y1'][i]
    right = Filter_dfval['x2'][i]
    bottom = Filter_dfval['y2'][i]
    f_image = f_image.crop((left, top, right, bottom))
    f_image = f_image.resize((200, 200))
#    list_img.append([Filter_dfval.index[i],f_image])
    foldercheck = Filter_dfval.index[i].split('/')[0]
    foldercheck1 = Filter_dfval.index[i].split('/')[1]
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/FULL_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/FULL_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/UPPER_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/UPPER_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/LOWER_BODY/' + foldercheck):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/LOWER_BODY/' + foldercheck)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/FULL_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/FULL_BODY/' + foldercheck+'/'+ foldercheck1)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/UPPER_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/UPPER_BODY/' + foldercheck+'/'+ foldercheck1)
    if not os.path.exists('DATA/DeepFashion_crop_cate1/img/val/LOWER_BODY/' + foldercheck+'/'+foldercheck1):
      os.mkdir('DATA/DeepFashion_crop_cate1/img/val/LOWER_BODY/' + foldercheck+'/'+ foldercheck1)
    f_image.save('DATA/DeepFashion_crop_cate1/img/val/'+Filter_dfval.img_cate1_name[i]+'/'+Filter_dfval.index[i])
print('Finish')