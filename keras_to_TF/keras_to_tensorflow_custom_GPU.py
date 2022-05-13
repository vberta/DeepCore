from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "_1"
## moved earlier
## try this
##from keras.backend import tensorflow_backend as K
##import tensorflow as tf
from  matplotlib import pyplot as plt
from array import array as array2
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.initializers import *
from keras.layers import AlphaDropout
from keras.layers import Input, LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Conv2DTranspose, concatenate, Concatenate, ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.models import Model,load_model, Sequential



from keras.optimizers import *
from numpy import concatenate as concatenatenp
from pathlib import Path
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import argparse
import argparse
import glob
import gzip
import keras
import math
import matplotlib as mpl
import matplotlib.backends.backend_pdf as backpdf
import numpy as np
import pickle
import pylab
import random
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import uproot
import uproot
import uproot
import pdb
#from keras import backend as K
# import keras
# from tensorflow.python.framework import ops


tf.compat.v1.disable_eager_execution()

weight_file_path = sys.argv[1]

def _to_tensor(x, dtype):
    return ops.convert_to_tensor(x, dtype=dtype)

def loss_mse_select_clipped(y_true, y_pred) :
    wei = y_true[:,:,:,:,-1:]
    pred = y_pred[:,:,:,:,:-1]
    true =  y_true[:,:,:,:,:-1]
    out =K.square(tf.clip_by_value(pred-true,-5,5))*wei
    return tf.reduce_sum(out, axis=None)/(tf.reduce_sum(wei,axis=None)*4) #4=numPar

def loss_ROI_crossentropy(target, output):
    epsilon_ = _to_tensor(keras.backend.epsilon(), output.dtype.base_dtype)
    # epsilon_ = keras.backend.epsilon()
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    # print("target shape=", target.shape)
    wei = target[:,:,:,:,-1:]
    target = target[:,:,:,:,:-1]
    # print("target shape=", target.shape, ", wei shape=", wei.shape, "output shape ", output.shape)
    output = output[:,:,:,:,:-1]
    output = math_ops.log(output / (1 - output))
    retval = nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=10)#900=works #2900=200x200, 125=30x30
    retval = retval*wei
    # print("output shape ", retval.shape)
    return tf.reduce_sum(retval, axis=None)/(tf.reduce_sum(wei,axis=None))

def loss_ROIsoft_crossentropy(target, output):
    epsilon_ = _to_tensor(keras.backend.epsilon(), output.dtype.base_dtype)
    # epsilon_ = keras.backend.epsilon()
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    # print("target shape=", target.shape)
    wei = target[:,:,:,:,-1:]
    target = target[:,:,:,:,:-1]
    # print("target shape=", target.shape, ", wei shape=", wei.shape, "output shape ", output.shape)
    output = output[:,:,:,:,:-1]
    output = math_ops.log(output / (1 - output))
    retval = nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=10)#900=works #2900=200x200, 125=30x30
    retval = retval*(wei+0.01)
    # print("output xentr ", retval)
    return tf.reduce_sum(retval, axis=None)/(tf.reduce_sum(wei,axis=None))

K.set_learning_phase(0)
K.set_image_data_format('channels_last')

try:
    ##net_model = load_model(weight_file_path,custom_objects={'loss_mse_select_clipped':loss_mse_select_clipped,'loss_ROIsoft_crossentropy':loss_ROI_crossentropy, '_to_tensor':_to_tensor})
    #net_model = load_model(weight_file_path,custom_objects={'loss_mse_select_clipped':loss_mse_select_clipped,'loss_ROI_crossentropy':loss_ROI_crossentropy, '_to_tensor':_to_tensor})
    net_model = tf.compat.v1.keras.models.load_model(weight_file_path,custom_objects={'loss_mse_select_clipped':loss_mse_select_clipped,'loss_ROI_crossentropy':loss_ROI_crossentropy, '_to_tensor':_to_tensor})
    #pdb.set_trace()
except ValueError as err:
    print('''Input file specified ({}) only holds the weights, and not the model defenition.
    Save the model using mode.save(filename.h5) which will contain the network architecture
    as well as its weights.
    If the model is saved using model.save_weights(filename.h5), the model architecture is
    expected to be saved separately in a json format and loaded prior to loading the weights.
    Check the keras documentation for more details (https://keras.io/getting-started/faq/)'''
          .format(weight_file_path))
    raise err

#net_model.save('test')
num_output = 2
pred = [None]*num_output
#pred_node_names = [None]*num_output
#for i in range(num_output):
#    pred_node_names[i] = "output_node"+str(i)

#outputNodeNames = ["reshape_1/Reshape:0","reshape_2/Reshape:0"]
outputNodeNames = ["reshape_1_1","reshape_2_1"]
#outputNodeNames = ["reshape_1/Reshape","reshape_2/Reshape"]
for i,outputNodeName in enumerate(outputNodeNames):
    #pdb.set_trace()
    print(i,outputNodeName)
    pred[i] = tf.identity(net_model.outputs[i], name=outputNodeName)
#pdb.set_trace()
print('output nodes names are: ', outputNodeNames)


#sess = K.get_session()
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

#if args.graph_def:
#    f = args.output_graphdef_file
#    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
#    print('saved the graph definition in ascii format at: ', str(Path(output_fld) / f))


# convert variables to constants and save

# In[ ]:

##from tensorflow.python.framework import graph_util
##from tensorflow.python.framework import graph_io
# tf.compat.v1.graph_util.convert_variables_to_constants new version
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import graph_io

#constant_graph = tf.compat.v1.graph_util.extract_sub_graph(sess.graph.as_graph_def(), pred_node_names)
#pdb.set_trace()
print(sess.graph.as_graph_def())
output_fld = "pb_files"
output_name = sys.argv[1].replace(".h5",".pb")
constant_graph = tf.compat.v1.graph_util.extract_sub_graph(sess.graph.as_graph_def(), outputNodeNames)
graph_io.write_graph(constant_graph, output_fld, output_name, as_text=False)
print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / output_name))
