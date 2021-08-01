from __future__ import print_function
import os
# os.environ['MKL_NUM_THREADS'] = '40'
# os.environ['GOTO_NUM_THREADS'] = '40'
# os.environ['OMP_NUM_THREADS'] = '40'
# os.environ['openmp'] = 'True'


# from ROOT import *
# from keras.callbacks import Callback
# from keras.models import Model,load_model, Sequential
# from keras.layers import Input, LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Conv2DTranspose, concatenate, Concatenate, ZeroPadding2D, UpSampling2D, UpSampling1D
# from keras.optimizers import *
# from keras.initializers import *
import numpy as np
# import tensorflow as tf
# from keras.backend import tensorflow_backend as K
# import keras
import math
import sys
import argparse
import matplotlib as mpl
mpl.use('Agg')

# import matplotlib.backends.backend_pdf as backpdf

# from keras.callbacks import ModelCheckpoint

from  matplotlib import pyplot as plt
import pylab
import glob

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import clip_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn

from numpy import concatenate as concatenatenp



loss_tot, loss_par, loss_prob, loss_val_tot, loss_val_par, loss_val_prob,= pylab.loadtxt('loss_all.txt', unpack = True)

# fig_tot=plt.figure(1000)
plt.figure(1000)
pylab.plot(loss_val_tot, color='darkorange', linewidth=2)
pylab.plot(loss_tot, color='royalblue', linewidth=2)
# plt.semilogy(loss_tot)
# plt.semilogy(loss_val_tot)
pylab.title('model loss', fontsize=24)
pylab.ylabel('loss', fontsize=22)
pylab.xlabel('epoch', fontsize=22)
# pylab.ylim(0.61,0.75)#1.9)
# pylab.ylim(0.82,1.1)#1.1 barrel version
pylab.ylim(2.4,3)#1.1
plt.grid(True)
pylab.legend(['test', 'train'], loc='upper right')
# plot = fig_tot.add_subplot(1, 1, 1)
# plot.set_yscale('log')
# plt.yscale('linear')
plt.text(4,1.08, "CMS ", weight='bold', size=17)
plt.text(4,1.065, "Preliminary Simulation", style='italic', size=14)
plt.text(4,1.05,"13 TeV ", size=14)
plt.text(4,1.035, r'QCD events ($\langle PU \rangle=35$), 1.8 TeV$<P_T^{jet}<2.4$ TeV',size=14)

pylab.savefig("loss_tot.pdf")
pylab.savefig("loss_tot.png")


plt.figure(1001)
pylab.plot(loss_val_par, color='darkorange', linewidth=2)
pylab.plot(loss_par, color='royalblue', linewidth=2)
pylab.title('model loss (parameters)', fontsize=24)
pylab.ylabel('loss', fontsize=22)
pylab.xlabel('epoch', fontsize=22)
# pylab.ylim(0.37,0.43)#1.1)
pylab.ylim(0.345,0.48)
plt.grid(True)
pylab.legend(['test', 'train'], loc='upper right')
plt.text(4,0.47, "CMS ", weight='bold', size=17)
plt.text(4,0.462, "Preliminary Simulation", style='italic', size=14)
plt.text(4,0.454, "13 TeV ", size=14)
plt.text(4,0.446, r'QCD events ($\langle PU \rangle=35$), 1.8 TeV$<P_T^{jet}<2.4$ TeV',size=14)

pylab.savefig("loss_par.pdf")
pylab.savefig("loss_par.png")


plt.figure(1002)
pylab.plot(loss_val_prob, color='darkorange', linewidth=2)
pylab.plot(loss_prob, color='royalblue', linewidth=2)
pylab.title('model loss (probability)', fontsize=24)
pylab.ylabel('loss', fontsize=22)
pylab.xlabel('epoch', fontsize=22)
# pylab.ylim(0.24,0.31)#1.2)
pylab.ylim(0.47,0.61)
plt.grid(True)
pylab.legend(['test', 'train'], loc='upper right')
plt.text(4,0.60, "CMS ", weight='bold', size=17)
plt.text(4,0.593, "Preliminary Simulation", style='italic', size=14)
plt.text(4,0.586,"13 TeV ", size=14)
plt.text(4,0.579, r'QCD events ($\langle PU \rangle=35$), 1.8 TeV$<P_T^{jet}<2.4$ TeV',size=14)

pylab.savefig("loss_prob.pdf")
pylab.savefig("loss_prob.png")
