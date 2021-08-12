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
base_path = 'DATA/Deepfashion_unzip_new/img'
import cv2
import numpy as np
from PIL import Image
import random
import PIL.Image
import tensorflow
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
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,AveragePooling2D,Dropout,Activation,Input,Flatten,Lambda
from tensorflow.keras.optimizers import SGD,Adam,RMSprop

import tensorflow.keras.applications.efficientnet as efn
from keras.utils.data_utils import get_file


def find_category_code (category_name, category_code_list):
  for i in category_code_list:
    if category_name == i[0]:
      return i[1]
  return null 

with    open('DATA/list_category_cloth.txt', 'r') as category_cloth_file, \
          open('DATA/train.txt', 'r') as train_filename, \
          open('DATA/train_attr.txt', 'r') as train_attr, \
          open('DATA/train_bbox.txt', 'r') as train_bbox, \
          open('DATA/train_cate.txt', 'r') as train_cate, \
          open('DATA/test.txt', 'r') as test_filename, \
          open('DATA/test_attr.txt', 'r') as test_attr, \
          open('DATA/test_bbox.txt', 'r') as test_bbox, \
          open('DATA/test_cate.txt', 'r') as test_cate, \
            open('DATA/val.txt', 'r') as val_filename, \
            open('DATA/val_attr.txt', 'r') as val_attr, \
            open('DATA/val_bbox.txt', 'r') as val_bbox, \
             open('DATA/val_cate.txt', 'r') as val_cate:
        list_category_cloth = [line.rstrip('\n') for line in category_cloth_file][2:]
        list_train_filename = [line.rstrip('\n') for line in train_filename]
        list_train_attr = [line.rstrip('\n') for line in train_attr]
        list_test_filename = [line.rstrip('\n') for line in test_filename]
        list_test_attr = [line.rstrip('\n') for line in test_attr]
        list_val_filename = [line.rstrip('\n') for line in val_filename]
        list_val_attr = [line.rstrip('\n') for line in val_attr]
        
        list_train_bbox = [line.rstrip('\n') for line in train_bbox]
        list_train_cate = [line.rstrip('\n') for line in train_cate]
        list_val_bbox = [line.rstrip('\n') for line in val_bbox]
        list_val_cate = [line.rstrip('\n') for line in val_cate]
        list_test_bbox = [line.rstrip('\n') for line in test_bbox]
        list_test_cate = [line.rstrip('\n') for line in test_cate]

        list_category_cloth = [splitter.split(line) for line in list_category_cloth]
        list_train_filename = [splitter.split(line) for line in list_train_filename]
        list_train_attr = [splitter.split(line) for line in list_train_attr]
        list_test_filename = [splitter.split(line) for line in list_test_filename]
        list_test_attr = [splitter.split(line) for line in list_test_attr]                         
        list_val_filename = [splitter.split(line) for line in list_val_filename]
        list_val_attr = [splitter.split(line) for line in list_val_attr]

        list_train_bbox = [splitter.split(line) for line in list_train_bbox]
        list_train_cate = [splitter.split(line) for line in list_train_cate]
        list_test_bbox = [splitter.split(line) for line in list_test_bbox]
        list_test_cate = [splitter.split(line) for line in list_test_cate]                          
        list_val_bbox = [splitter.split(line) for line in list_val_bbox]
        list_val_cate = [splitter.split(line) for line in list_val_cate]        
        list_train1 =[(a[0],
                      a[0].split('/')[1].split('_')[-1],find_category_code(a[0].split('/')[1].split('_')[-1],list_category_cloth), 
                      d[0],int(c[0]),int(c[1]),int(c[2]),int(c[3]),                      
                      b[0], b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],
                      b[10], b[11],b[12],b[13],b[14],b[15],b[16],b[17],b[18],b[19],
                      b[20], b[21],b[22],b[23],b[24],b[25]) 
                      for a, b,c,d in zip (list_train_filename, list_train_attr, list_train_bbox, list_train_cate)]
        list_val1 =[(a[0],
                      a[0].split('/')[1].split('_')[-1], find_category_code(a[0].split('/')[1].split('_')[-1],list_category_cloth),
                      d[0],int(c[0]),int(c[1]),int(c[2]),int(c[3]),                        
                      b[0], b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],
                      b[10], b[11],b[12],b[13],b[14],b[15],b[16],b[17],b[18],b[19],
                      b[20], b[21],b[22],b[23],b[24],b[25]) 
                      for a, b,c,d in zip (list_val_filename, list_val_attr, list_val_bbox, list_val_cate)]
        list_test1 =[(a[0],
                      a[0].split('/')[1].split('_')[-1], find_category_code(a[0].split('/')[1].split('_')[-1],list_category_cloth),
                      d[0],int(c[0]),int(c[1]),int(c[2]),int(c[3]),                        
                      b[0], b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],
                      b[10], b[11],b[12],b[13],b[14],b[15],b[16],b[17],b[18],b[19],
                      b[20], b[21],b[22],b[23],b[24],b[25]) 
                      for a, b,c,d in zip (list_test_filename, list_test_attr, list_test_bbox, list_test_cate)]
                      
                      
#DF_attributes
a = pd.DataFrame(list_train1)
b = pd.DataFrame(list_val1)
c = pd.DataFrame(list_test1)
print(len(a))
print(len(b))
print(len(c))

a =a.set_index(a.columns[0])
b=b.set_index(b.columns[0])
c =c.set_index(c.columns[0])

#DF_attributes
def filter_dict_bboxes_train():
  dict_train1 =[]
  for row in a.index:
    if os.path.exists(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row))):
      dict_train1.append(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row)))
  return dict_train1
def filter_dict_bboxes_val():
  dict_val1 =[]
  for row in b.index:
    if os.path.exists(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row))):
      dict_val1.append(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row)))
  return dict_val1
def filter_dict_bboxes_test():
  dict_test1 =[]
  for row in c.index:
    if os.path.exists(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row))):
      dict_test1.append(os.path.join(os.path.join('DATA/Deepfashion_unzip_new', row)))
  return dict_test1
  
dict_train1 =filter_dict_bboxes_train()  
dict_val1 =filter_dict_bboxes_val()
dict_test1 =filter_dict_bboxes_test()
print(len(dict_train1))
print(len(dict_val1))
print(len(dict_test1))
#df_attributes
def filter2_train():
  dict_train2 =[]
  for row in dict_train1:
    row =row.split('unzip_new/')[-1]
    dict_train2.append(row)
  return dict_train2
def filter2_val():
  dict_val2 =[]
  for row in dict_val1:
    row =row.split('unzip_new/')[-1]
    dict_val2.append(row)
  return dict_val2
def filter2_test():
  dict_test2 =[]
  for row in dict_test1:
    row =row.split('unzip_new/')[-1]
    dict_test2.append(row)
  return dict_test2
  
dict_train2 =filter2_train()
dict_val2 =filter2_val()
dict_test2 =filter2_test()

Filter_dftrain  = a[a.index.isin(dict_train2)]
Filter_dfval  = b[b.index.isin(dict_val2)]
Filter_dftest  = c[c.index.isin(dict_test2)]

print(len(Filter_dftrain))
print(len(Filter_dfval))
print(len(Filter_dftest))


Filter_dftrain.columns =['cate_name','cate_num_lv1','cate_num_lv2','x1','y1','x2','y2','floral', 'graphic','striped','embroidered','pleated','solid','lattice','long_sleeve','short_sleeve','sleeveless','maxi_length','mini_length','no_dress','crew_neckline','v_neckline','square_neckline','no_neckline','denim','chiffon','cotton','leather','faux','knit','tight','loose','conventional']
Filter_dfval.columns =['cate_name','cate_num_lv1','cate_num_lv2','x1','y1','x2','y2','floral', 'graphic','striped','embroidered','pleated','solid','lattice','long_sleeve','short_sleeve','sleeveless','maxi_length','mini_length','no_dress','crew_neckline','v_neckline','square_neckline','no_neckline','denim','chiffon','cotton','leather','faux','knit','tight','loose','conventional']
Filter_dftest.columns =['cate_name','cate_num_lv1','cate_num_lv2','x1','y1','x2','y2','floral', 'graphic','striped','embroidered','pleated','solid','lattice','long_sleeve','short_sleeve','sleeveless','maxi_length','mini_length','no_dress','crew_neckline','v_neckline','square_neckline','no_neckline','denim','chiffon','cotton','leather','faux','knit','tight','loose','conventional']


Filter_dftrain.iloc[:,2:]=Filter_dftrain.iloc[:,2:].astype(int)
Filter_dfval.iloc[:,2:]=Filter_dfval.iloc[:,2:].astype(int)
Filter_dftest.iloc[:,2:]=Filter_dftest.iloc[:,2:].astype(int)


y_train_1 =Filter_dftrain.iloc[:,7:14]
y_val_1 =Filter_dfval.iloc[:,7:14]
y_train_2 =Filter_dftrain.iloc[:,14:17]
y_val_2 =Filter_dfval.iloc[:,14:17]
y_train_3 =Filter_dftrain.iloc[:,17:20]
y_val_3 =Filter_dfval.iloc[:,17:20]
y_train_4 =Filter_dftrain.iloc[:,20:24]
y_val_4 =Filter_dfval.iloc[:,20:24]
y_train_5 =Filter_dftrain.iloc[:,24:30]
y_val_5 =Filter_dfval.iloc[:,24:30]
y_train_6 =Filter_dftrain.iloc[:,30:33]
y_val_6 =Filter_dfval.iloc[:,30:33]

print(Filter_dftrain.index[0])
# Attributes_crop image around bouding box-Filter_dftrain
list_img =[]
for i in range(0, len(Filter_dftrain)):
  f_image = Image.open('DATA/Deepfashion_unzip_new/'+Filter_dftrain.index[i])
  left = Filter_dftrain['x1'][i]
  top = Filter_dftrain['y1'][i]
  right = Filter_dftrain['x2'][i]
  bottom = Filter_dftrain['y2'][i]


  f_image = f_image.crop((left, top, right, bottom))
  f_image = f_image.resize((150, 150))
  list_img.append([Filter_dftrain.index[i],f_image])
  
list_img_val =[]
for i in range(0, len(Filter_dfval)):
  f_image = Image.open('DATA/Deepfashion_unzip_new/'+Filter_dfval.index[i])
  left = Filter_dfval['x1'][i]
  top = Filter_dfval['y1'][i]
  right = Filter_dfval['x2'][i]
  bottom = Filter_dftrain['y2'][i]


  f_image = f_image.crop((left, top, right, bottom))
  f_image = f_image.resize((150, 150))
  list_img_val.append([Filter_dfval.index[i],f_image])


train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_image =[]
for i in range(0, len(Filter_dftrain)):
            img = list_img[i][1]
            x = image.img_to_array(img)
            x = train_datagen.random_transform(x)
            x = train_datagen.standardize(x)
            x =x/255
            train_image.append(x)


val_image =[]
for i in range(0, len(Filter_dfval)):
            img = list_img_val[i][1]
            x = image.img_to_array(img)
            x = train_datagen.random_transform(x)
            x = train_datagen.standardize(x)
            x =x/255
            val_image.append(x)
# import pickle
# outfile = open("DAO/output/train_image_14000.txt","wb+")
# pickle.dump(train_image,outfile)

# outfile1 = open("DAO/output/val_image_40000.txt","wb+")
# pickle.dump(val_image,outfile1)

x_train = np.array(train_image)
x_val = np.array(val_image)
print(x_train.shape, y_train_1.shape, y_train_2.shape, y_train_3.shape, y_train_4.shape, y_train_5.shape, y_train_6.shape, x_val.shape, y_val_1.shape, y_val_2.shape, y_val_3.shape, y_val_4.shape, y_val_5.shape, y_val_6.shape)


import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


# def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  # """
  # Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  # Some examples of different formats of class_series and their outputs are:
    # - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    # {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    # - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    # {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    # - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    # {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    # - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    # {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  # The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  # of appareance of the label when the dataset was processed. 
  # In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  # Author: Angel Igareta (angel@igareta.com)
  # """
  # if multi_class:
    # # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    # if one_hot_encoded:
      # class_series = np.argmax(class_series, axis=1)
  
    # # Compute class weights with sklearn method
    # class_labels = np.unique(class_series)
    # class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    # return dict(zip(class_labels, class_weights))
  # else:
    # # It is neccessary that the multi-label values are one-hot encoded
    # mlb = None
    # if not one_hot_encoded:
      # mlb = MultiLabelBinarizer()
      # class_series = mlb.fit_transform(class_series)

    # n_samples = len(class_series)
    # n_classes = len(class_series[0])

    # # Count each class frequency
    # class_count = [0] * n_classes
    # for classes in class_series:
        # for index in range(n_classes):
            # if classes[index] != 0:
                # class_count[index] += 1
    
    # # Compute class weights using balanced method
    # class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    # class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    # return dict(zip(class_labels, class_weights))

# y_integers1 = np.argmax(y_train_1, axis=1)

# y_integers2 = np.argmax(y_train_2, axis=1)

# y_integers3 = np.argmax(y_train_3, axis=1)

# y_integers4 = np.argmax(y_train_4, axis=1)

# y_integers5 = np.argmax(y_train_5, axis=1)

# y_integers6 = np.argmax(y_train_6, axis=1)


# class_weights1 =generate_class_weights(y_integers1)
# class_weights2 =generate_class_weights(y_integers2)
# class_weights3 =generate_class_weights(y_integers3)
# class_weights4 =generate_class_weights(y_integers4)
# class_weights5 =generate_class_weights(y_integers5)
# class_weights6 =generate_class_weights(y_integers6)

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
# Instantiate the label encoder
le = LabelEncoder()
y_classes1 = y_train_1.idxmax(1, skipna=False)
# Fit the label encoder to our label series
le.fit(list(y_classes1))
# Create integer based labels Series
y_integers1 = le.transform(list(y_classes1))
# Create dict of labels : integer representation
labels_and_integers1 = dict(zip(y_classes1, y_integers1))
class_weights1 = compute_class_weight('balanced', np.unique(y_integers1), y_integers1)
class_weights_dict1 = dict(zip(le.transform(list(le.classes_)), class_weights1))


y_classes2 = y_train_2.idxmax(1, skipna=False)
le.fit(list(y_classes2))
y_integers2 = le.transform(list(y_classes2))
labels_and_integers2 = dict(zip(y_classes2, y_integers2))
class_weights2 = compute_class_weight('balanced', np.unique(y_integers2), y_integers2)
class_weights_dict2 = dict(zip(le.transform(list(le.classes_)), class_weights2))

y_classes3= y_train_3.idxmax(1, skipna=False)
le.fit(list(y_classes3))
y_integers3 = le.transform(list(y_classes3))
labels_and_integers3 = dict(zip(y_classes3, y_integers3))
class_weights3 = compute_class_weight('balanced', np.unique(y_integers3), y_integers3)
class_weights_dict3 = dict(zip(le.transform(list(le.classes_)), class_weights3))

y_classes4= y_train_4.idxmax(1, skipna=False)
le.fit(list(y_classes4))
y_integers4 = le.transform(list(y_classes4))
labels_and_integers4 = dict(zip(y_classes4, y_integers4))
class_weights4 = compute_class_weight('balanced', np.unique(y_integers4), y_integers4)
class_weights_dict4 = dict(zip(le.transform(list(le.classes_)), class_weights4))

y_classes5= y_train_5.idxmax(1, skipna=False)
le.fit(list(y_classes5))
y_integers5 = le.transform(list(y_classes5))
labels_and_integers5 = dict(zip(y_classes5, y_integers5))
class_weights5 = compute_class_weight('balanced', np.unique(y_integers5), y_integers5)
class_weights_dict5 = dict(zip(le.transform(list(le.classes_)), class_weights5))

y_classes6= y_train_6.idxmax(1, skipna=False)
le.fit(list(y_classes6))
y_integers6 = le.transform(list(y_classes6))
labels_and_integers6 = dict(zip(y_classes6, y_integers6))
class_weights6 = compute_class_weight('balanced', np.unique(y_integers6), y_integers6)
class_weights_dict6 = dict(zip(le.transform(list(le.classes_)), class_weights6))

print(class_weights_dict6)
print(class_weights_dict5)
print(class_weights_dict4)
print(class_weights_dict3)
print(class_weights_dict2)
print(class_weights_dict1)

def weighted_categorical_crossentropy(class_weight):
    def loss(y_obs,y_pred):
        y_obs =tensorflow.dtypes.cast(y_obs,tensorflow.int32)
        hothot =tensorflow.one_hot(tensorflow.reshape(y_obs,[-1]),depth =len(class_weight))
        weight=tensorflow.math.multiply(class_weight,hothot)
        weight=tensorflow.reduce_sum(weight,axis =-1)
        losses =tensorflow.compat.v1.losses.sparse_softmax_cross_entropy(labels =y_obs,logits =y_pred,weights=weight)
        return losses
    return loss

#losses =tensorflow.compat.v1.losses.sparse_softmax_cross_entropy(labels =y_obs,logits =y_pred,weights=weight)

loss1 =weighted_categorical_crossentropy(list(class_weights_dict1.values()))
loss2 =weighted_categorical_crossentropy(list(class_weights_dict2.values()))
loss3 =weighted_categorical_crossentropy(list(class_weights_dict3.values()))
loss4 =weighted_categorical_crossentropy(list(class_weights_dict4.values()))
loss5 =weighted_categorical_crossentropy(list(class_weights_dict5.values()))
loss6 =weighted_categorical_crossentropy(list(class_weights_dict6.values()))    

print(loss1)
print(loss2)
print(loss3)
print(loss4)
print(loss5)
print(loss6)
# from sklearn.utils import class_weight
# class_weight1 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_1) ,y_train_1)
# class_weight2 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_2) ,y_train_2)
# class_weight3 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_3) ,y_train_3)
# class_weight4 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_4) ,y_train_4)
# class_weight5 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_5) ,y_train_5)
# class_weight6 = class_weight.compute_class_weight('balanced' ,np.unique(y_train_6) ,y_train_6)
# print(class_weight1, class_weight2,class_weight3, class_weight4, class_weight5, class_weight6)
# conv_base = efn.EfficientNetB7(input_shape = (200, 200, 3), include_top = False, weights = 'imagenet')
# for layer in conv_base.layers:
    # layer.trainable = False
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3

input_images = Input(shape=(150, 150, 3), dtype='float32', name='images')
conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False
texture_model = Sequential()
texture_model.add(conv_base)
texture_model.add(Flatten())
texture_model.add(Dropout(0.25))
texture_model.add(Dense(256, activation='relu'))
texture_model.add(Dropout(0.25))
texture_model.add(Dense(7, activation='softmax'))
texture_model._name = 'texture_dao'
texture_output = texture_model(input_images)


sleeve_model = Sequential()
sleeve_model.add(conv_base)
sleeve_model.add(Flatten())
sleeve_model.add(Dropout(0.25))
sleeve_model.add(Dense(256, activation='relu'))
sleeve_model.add(Dropout(0.25))
sleeve_model.add(Dense(3, activation='softmax'))
sleeve_model._name = 'sleeve_dao'
sleeve_output = sleeve_model(input_images)

length_model = Sequential()
length_model.add(conv_base)
length_model.add(Flatten())
length_model.add(Dropout(0.25))
length_model.add(Dense(256, activation='relu'))
length_model.add(Dropout(0.25))
length_model.add(Dense(3, activation='softmax'))
length_model._name = 'length_dao'
length_output = length_model(input_images)

neckline_model = Sequential()
neckline_model.add(conv_base)
neckline_model.add(Flatten())
neckline_model.add(Dropout(0.25))
neckline_model.add(Dense(256, activation='relu'))
neckline_model.add(Dropout(0.25))
neckline_model.add(Dense(4, activation='softmax'))
neckline_model._name = 'neckline_dao'
neckline_output = neckline_model(input_images)

fabric_model = Sequential()
fabric_model.add(conv_base)
fabric_model.add(Flatten())
fabric_model.add(Dropout(0.25))
fabric_model.add(Dense(256, activation='relu'))
fabric_model.add(Dropout(0.25))
fabric_model.add(Dense(6, activation='softmax'))
fabric_model._name = 'fabric_dao'
fabric_output = fabric_model(input_images)

style_model = Sequential()
style_model.add(conv_base)
style_model.add(Flatten())
style_model.add(Dropout(0.25))
style_model.add(Dense(256, activation='relu'))
style_model.add(Dropout(0.25))
style_model.add(Dense(3, activation='softmax'))
style_model._name = 'style_dao'
style_output = style_model(input_images)

model = Model(input_images, [texture_output, sleeve_output,length_output,neckline_output,fabric_output,style_output])

# model.compile(loss={'texture_dao': 'categorical_crossentropy', 'sleeve_dao': 'categorical_crossentropy','length_dao': 'categorical_crossentropy', 'neckline_dao': 'categorical_crossentropy','fabric_dao': 'categorical_crossentropy', 'style_dao': 'categorical_crossentropy'},
              # optimizer=RMSprop(learning_rate=1e-4),
              # metrics=['accuracy'])
              
model.compile(loss={'texture_dao': loss1, 'sleeve_dao': loss2,'length_dao': loss3, 'neckline_dao': loss4,'fabric_dao': loss5, 'style_dao': loss6},
              optimizer=RMSprop(learning_rate=1e-4),
              metrics={'texture_dao': 'accuracy', 'sleeve_dao': 'accuracy','length_dao': 'accuracy', 'neckline_dao': 'accuracy','fabric_dao': 'accuracy', 'style_dao': 'accuracy'})
model.summary()

#model.load_weights('code/output/model_attr26_v1.hdf5')
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)
cb_checkpointer = ModelCheckpoint(filepath = 'DAO/output/model_attr26_20k_mix_InceptionV3.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) ]:
model.fit(x_train, {"texture_dao": y_train_1, "sleeve_dao": y_train_2,"length_dao": y_train_3, "neckline_dao": y_train_4,"fabric_dao": y_train_5, "style_dao": y_train_6}, epochs=50, validation_data=(x_val, {"texture_dao": y_val_1, "sleeve_dao": y_val_2,"length_dao": y_val_3, "neckline_dao": y_val_4,"fabric_dao": y_val_5, "style_dao": y_val_6}), batch_size=64,callbacks=[lr_reducer,cb_checkpointer, cb_early_stopper,tensorboard])
model.load_weights('DAO/output/model_attr26_20k_mix_InceptionV3.hdf5')
model.save('DAO/output/model_attr26_20k_mix_InceptionV3.h5')