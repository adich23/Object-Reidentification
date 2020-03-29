from sklearn import metrics
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model,Sequential
from keras import optimizers
from keras.regularizers import l2
from keras.applications import resnet50
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config_2 import *
from math import ceil
import json
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from collections import Counter
from keras import backend as K
from keras.layers import *
from keras.models import Model, load_model
import string
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from keras.optimizers import Adam

# global constants
DIM_ORDERING = 'tf'
CONCAT_AXIS = -1

def vgg_original (input_shape):
    input1 = Input(input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    return Model(input1,x)

def small_vgg_car(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(input1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)

    return Model(input1,x)

def small_vgg_plate(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)

    return Model(input1,x)

#------------------------------------------------------------------------------
def get_batch_inds(batch_size, idx, N):
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 >= N:
            idx1 = N
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds
#------------------------------------------------------------------------------
def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1, [0,1])
    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, num_test_steps, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(num_test_steps):
        X, Y, paths = next(test_gen)
        Y_ = model.predict(X)
        for y1, yreg, y2, p0, p1 in zip(Y_[0].tolist(), Y_[1].tolist(), Y['class_output'].argmax(axis=-1).tolist(), paths[0], paths[1]):
          y1_class = np.argmax(y1)
          ypred.append(y1_class)
          ytrue.append(y2)
          a.write("%s;%s;%d;%d;%f;%s\n" % (p0, p1, y2, y1_class, yreg[0], str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()
#------------------------------------------------------------------------------
def process_load(f1, vec_size):
    _i1 = image.load_img(f1, target_size=vec_size)
    _i1 = image.img_to_array(_i1, dtype='uint8')
    return _i1

#------------------------------------------------------------------------------
def load_img(img, vec_size, vec_size2, metadata_dict):
  iplt0 = process_load(img[0][0], vec_size)
  iplt1 = process_load(img[2][0], vec_size)
  iplt2 = process_load(img[1][0], vec_size2)
  iplt3 = process_load(img[3][0], vec_size2)

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":img[0][0],
        "p2":img[2][0],
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }
  if metadata_dict is not None:
    diff = abs(np.array(metadata_dict[img[0][0]][:7]) - np.array(metadata_dict[img[2][0]][:7])).tolist()
    for i in range(len(diff)):
      diff[i] = 1 if diff[i] else 0
    d1['metadata'] = np.array(metadata_dict[img[0][0]] + metadata_dict[img[2][0]] + diff)

  return d1

#------------------------------------------------------------------------------
def generator(features, batch_size, executor, vec_size, vec_size2, type=None,metadata_dict=None, metadata_length=0, augmentation=False, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size = (len(inds),) + vec_size
      b1 = np.zeros(_vec_size)
      b2 = np.zeros(_vec_size)
      _vec_size2 = (len(inds),) + vec_size2
      b3 = np.zeros(_vec_size2)
      b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      c1 = []
      c2 = []
      if metadata_length>0:
        metadata = np.zeros((len(inds),metadata_length))

      futures = [executor.submit(partial(load_img, features[index], vec_size, vec_size2, metadata_dict)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        b1[i,:,:,:] = r['i0']
        b2[i,:,:,:] = r['i1']
        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        c1.append(r['c1'])
        c2.append(r['c2'])
        b3[i,:,:,:] = r['i2']
        b4[i,:,:,:] = r['i3']
        if metadata_length>0:
          metadata[i,:] = r['metadata']

      if augmentation:
        b1 = augs[0][0].augment_images(b1.astype('uint8')) / 255
        b2 = augs[1][0].augment_images(b2.astype('uint8')) / 255
        b3 = augs[2][0].augment_images(b3.astype('uint8')) / 255
        b4 = augs[3][0].augment_images(b4.astype('uint8')) / 255
      else:
        b1 = b1 / 255
        b2 = b2 / 255
        b3 = b3 / 255
        b4 = b4 / 255

      blabels2 = np.array(blabels)
      blabels = np_utils.to_categorical(blabels2, 2)
      y = {"class_output":blabels, "reg_output":blabels2}
      if type is None:
        result = [[b1, b2, b3, b4], y]
      elif type == 'plate':
        result = [[b1, b2], y]
      elif type == 'car':
        result = [[b3, b4], y]
      if metadata_length>0:
        result[0].append(metadata)
      if with_paths:
          result += [[p1,p2]]

      yield result

#------------------------------------------------------------------------------

def load_img_temporal(img, vec_size, vec_size2, tam, metadata_dict):
  iplt0 = [process_load(img[0][i], vec_size) for i in range(tam)]
  iplt1 = [process_load(img[2][i], vec_size) for i in range(tam)]
  iplt2 = [process_load(img[1][i], vec_size2) for i in range(tam)]
  iplt3 = [process_load(img[3][i], vec_size2) for i in range(tam)]

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "i2":iplt2,
        "i3":iplt3,
        "l":img[4],
        "p1":str(img[0]),
        "p2":str(img[2]),
        "c1":img[5]['color'],
        "c2":img[5]['color']
        }

  d1['metadata'] = []
  for i in range(tam):
    diff = abs(np.array(metadata_dict[img[0][i]][:7]) - np.array(metadata_dict[img[2][i]][:7])).tolist()
    for j in range(len(diff)):
      diff[j] = 1 if diff[j] else 0
    d1['metadata'] += metadata_dict[img[0][i]] + metadata_dict[img[2][i]] + diff
  d1['metadata'] = np.array(d1['metadata'])
  return d1
#------------------------------------------------------------------------------
def generator_temporal(features, batch_size, executor, vec_size, vec_size2, tam, metadata_dict, metadata_length, augmentation=False, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size = (len(inds),tam, ) + vec_size
      b1 = np.zeros(_vec_size)
      b2 = np.zeros(_vec_size)
      _vec_size2 = (len(inds),tam, ) + vec_size2
      b3 = np.zeros(_vec_size2)
      b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      c1 = []
      c2 = []
      metadata = np.zeros((len(inds),metadata_length))

      futures = [executor.submit(partial(load_img_temporal, features[index], vec_size, vec_size2, tam, metadata_dict)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        for j in range(tam):
          b1[i,j,:,:,:] = r['i0'][j]
          b2[i,j,:,:,:] = r['i1'][j]
          b3[i,j,:,:,:] = r['i2'][j]
          b4[i,j,:,:,:] = r['i3'][j]

        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        c1.append(r['c1'])
        c2.append(r['c2'])
        metadata[i,:] = r['metadata']

      if augmentation:
        for j in range(tam):
          b1[:,j,:] = augs[0][j].augment_images(b1[:,j,:].astype('uint8')) / 255
          b2[:,j,:] = augs[1][j].augment_images(b2[:,j,:].astype('uint8')) / 255
          b3[:,j,:] = augs[2][j].augment_images(b3[:,j,:].astype('uint8')) / 255
          b4[:,j,:] = augs[3][j].augment_images(b4[:,j,:].astype('uint8')) / 255
      else:
        for j in range(tam):
          b1[:,j,:] = b1[:,j,:] / 255
          b2[:,j,:] = b2[:,j,:] / 255
          b3[:,j,:] = b3[:,j,:] / 255
          b4[:,j,:] = b4[:,j,:] / 255

      blabels2 = np.array(blabels)
      blabels = np_utils.to_categorical(blabels2, 2)
      y = {"class_output":blabels, "reg_output":blabels2}
      result = [[b3, b4, metadata], y]

      if with_paths:
          result += [[p1,p2]]

      yield result
