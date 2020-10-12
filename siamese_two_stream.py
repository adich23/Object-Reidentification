from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config_2 import *
from math import ceil
import json
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Model, load_model
import string
import pandas as pd
from sys import argv
from custom_layers import *
from collections import Counter
import os
import argparse

#------------------------------------------------------------------------------
def siamese_model(input1, input2):
  left_input_P = Input(input1)
  right_input_P = Input(input1)
  left_input_C = Input(input2)
  right_input_C = Input(input2)
  # Select which CNN to choose for Surrounding stream
  convnet_plate = vgg_custom(input1)# small_vgg_plate(input1)
  encoded_l_P = convnet_plate(left_input_P)
  encoded_r_P = convnet_plate(right_input_P)
  convnet_car =  vgg_custom(input2)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)
  inputs = [left_input_P, right_input_P, left_input_C, right_input_C]

  # Add the distance function to the network
  L1_distanceP = L1_layer([encoded_l_P, encoded_r_P])
  L1_distanceC = L1_layer([encoded_l_C, encoded_r_C])
  concatL1 = Concatenate()([L1_distanceP, L1_distanceC])
  x = Dense(1024, activation='relu')(concatL1)
  x = Dense(1024, kernel_initializer='normal',activation='relu')(x)
  x = Dense(1024, kernel_initializer='normal',activation='relu')(x)
  predF2 = Dense(2,kernel_initializer='normal',activation='softmax', name='class_output')(x)
  regF2 = Dense(1,kernel_initializer='normal',activation='sigmoid', name='reg_output')(x)
  optimizer = Adam(0.0001)
  losses = {
     'class_output': 'binary_crossentropy',
     'reg_output': 'mean_squared_error'
  }

  lossWeights = {"class_output": 1.0, "reg_output": 1.0}

  model = Model(inputs=inputs, outputs=[predF2, regF2])
  model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer)
  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':
  # Trained models will be saved as - Set-0 model_two_stream_surr_0_car-vgg-96.h5, Set-1 model_two_stream_surrall_1_car-vgg-96.h5, ....
  # Validation output files saved as - For Set-0 - validation_two_stream_surr_0_car-vgg-96_inferences_output.txt
  parser = argparse.ArgumentParser()
  parser.add_argument("--car_size", help="Select Car image size from 64, 96, 128, 192",choices=[64,96,128,192],type=int, required=True)
  parser.add_argument("--surr_size", help="Select Car image size from 64, 96, 128, 192",choices=[64,96,128,192],type=int, required=True)
  parser.add_argument("--dataset", help="Data type - 3-sided or 4-sided ",type=str, default="_surr_10000")
  parser.add_argument("--suffix", help="Suffix for model name ",type=str, default="")
  parser.add_argument("--train", help="Train",action='store_true', default=False)
  parser.add_argument("--test", help="Train",action='store_true', default=False)
  parser.add_argument("--predict", help="Train",action='store_true', default=False)
  parser.add_argument("--model_folder", help="Saved Model Folder for Predict and Test",type=str)
  parser.add_argument("--json_path", help="JSON for Predict",type=str)
  args = parser.parse_args()
  

  if args.train:
    type1 = 'train'
  elif args.predict:
    type1 = 'predict'
  elif args.test:
    type1 = 'test'
  else:
    print("Either of Train or Predict or Test argument is required.")
    exit(1)
  

  PARAMS = {
      'car_size':args.car_size,
      'surr_size':args.surr_size,
      'dataset':args.dataset,
      'suffix':args.suffix,
      'dataset':args.dataset,
      'model':"-"+str(args.car_size)+"-"+str(args.surr_size)
  }
  
  if PARAMS['suffix'] == "":
    PARAMS['suffix'] = PARAMS['model']
  
#   suffix = "_surr_10000"
#   model_name = "-96"
  

  data = json.load(open('%s/dataset%s.json' % (path,PARAMS['dataset'])))
#   data = json.load(open('%s/dataset_surr_500.json' % (path)))

  keys = ['Set01','Set02','Set03','Set04','Set05']

  input1 = (PARAMS['surr_size'],PARAMS['surr_size'],nchannels)
  input2 = (PARAMS['car_size'],PARAMS['car_size'],nchannels)

  if type1=='train':
    for k in range(len(keys)):
      K.clear_session()
      val = data[keys[(k+2)%5]]
      aux = keys[:]
      print ("Valid: ",aux[(k+2)%5])
#       aux.pop((k+2)%5)
      trn = data[aux[k]] + data[aux[(k+1)%5]]
      print ("Train : ",aux[k]," ",aux[(k+1)%5],"\tTest: ",aux[(k+3)%5],aux[(k+4)%5])
      train_steps_per_epoch = ceil(len(trn) / batch_size)
      val_steps_per_epoch = ceil(len(val) / batch_size)

      ex1 = ProcessPoolExecutor(max_workers = 8)
      ex2 = ProcessPoolExecutor(max_workers = 8)

      trnGen = generator(trn, batch_size, ex1, input1, input2,  augmentation=True)
      tstGen = generator(val, batch_size, ex2, input1, input2)
      siamese_net = siamese_model(input1, input2)
      f1 = 'model_two_stream%s%s_%d.h5' % (PARAMS['model'],PARAMS['dataset'],k)
      #fit model
      history = siamese_net.fit_generator(trnGen,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=NUM_EPOCHS,
                                    validation_data=tstGen,
                                    validation_steps=val_steps_per_epoch)#,callbacks=[neptune_logger])

      #validate plate model
      tstGen2 = generator(val, batch_size, ex2, input1, input2, with_paths = True)
      test_report('validation_two_stream%s%s_%d' % (PARAMS['model'],PARAMS['dataset'],k),siamese_net, val_steps_per_epoch, tstGen2)

      siamese_net.save(f1)

  elif type1 == 'test':
#     folder = argv[2]
    if not args.model_folder:
        print("--model_folder argument is required for Predict or Test")
    folder = args.model_folder
    for k in range(len(keys)):
      K.clear_session()
      aux = keys[:]
#       aux.pop(k)
      print ("Train : ",aux[k]," ",aux[(k+1)%5],"\tTest: ",aux[(k+3)%5],aux[(k+4)%5])
      tst = data[aux[(k+3)%5]] + data[aux[(k+4)%5]]
      ex3 = ProcessPoolExecutor(max_workers = 12)
      tst_steps_per_epoch = ceil(len(tst) / batch_size)
      tstGen2 = generator(tst, batch_size, ex3, input1, input2, with_paths = True)
      f1 = os.path.join(folder,'model_two_stream_%d.h5' % (k))
      siamese_net = load_model(f1)
      test_report('test_two_stream%s_%d' % (PARAMS['suffix'],k),siamese_net, tst_steps_per_epoch, tstGen2)
  elif type1 == 'predict':

    results = []
    data = json.load(open(argv[2]))

    img1 = (process_load(data['img1_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img2 = (process_load(data['img2_plate'], input1)/255.0).reshape(1,input1[0],input1[1],input1[2])
    img3 = (process_load(data['img1_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])
    img4 = (process_load(data['img2_shape'], input2)/255.0).reshape(1,input2[0],input2[1],input2[2])

    X = [img1, img2, img3, img4]

#     folder = argv[3]
    if not args.model_folder:
        print("--model_folder argument is required for Predict or Test")
    folder = args.model_folder
    
    for k in range(len(keys)):
      K.clear_session()
      f1 = os.path.join(folder,'model_two_stream%s_%d.h5' % (PARAMS['suffix'],k))
      model = load_model(f1)
      Y_ = model.predict(X)
      results.append(np.argmax(Y_[0]))
      print("model %d: %s" % (k+1,"positive" if results[k]==POS else "negative"))
    print("final result: %s" % ("positive" if Counter(results).most_common(1)[0][0]==POS else "negative"))
    
