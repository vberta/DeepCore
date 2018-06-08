from __future__ import print_function
import os
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['GOTO_NUM_THREADS'] = '40'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['openmp'] = 'True'

from keras.callbacks import Callback
from keras.models import Model,load_model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Conv2DTranspose, concatenate, Concatenate, ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.optimizers import *
import numpy as np
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import keras
import math
import sys
import argparse
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.backends.backend_pdf as backpdf

from keras.callbacks import ModelCheckpoint

from  matplotlib import pyplot as plt
import pylab
import glob



#######################################
#
# USAGE: python yoloJet.py --seed SEED --convert --input INPUT --training --predict --output
#
#######################################


parser = argparse.ArgumentParser(description="Toy MC and NN for high pt jet clustering")
parser.add_argument(
    '--seed',
    dest='Seed',
    action='store',
    default="8",
    type=int,
    help='seed of simulator')
parser.add_argument(
    '--convert',
    dest='Convert',
    action='store_const',
    const=True,
    default=False,
    help='convert root to numpy array')
parser.add_argument(
    '--training',
    dest='Training',
    action='store_const',
    const=True,
    default=False,
    help='do the training of NN')
parser.add_argument(
    '--predict',
    dest='Predict',
    action='store_const',
    const=True,
    default=False,
    help='do the prediction of NN')
parser.add_argument(
    '--output',
    dest='Output',
    action='store_const',
    const=True,
    default=False,
    help='produce the output root file. NB: do not use on GPU')
parser.add_argument(
    '--input',
    dest='Input',
    action='store',
    default="toy_MC_1000.npy",
    type=str,
    help='name of the MC input file')


args = parser.parse_args()

seed=args.Seed
convert = args.Convert
output = args.Output
input_name = args.Input
train = args.Training
predict = args.Predict

with tf.Session(config=tf.ConfigProto(
                   intra_op_parallelism_threads=40)) as sess:
   K.set_session(sess)

class wHistory(keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs={}):
       if epoch % 10 == 0 :
               self.model.save("trained"+str(epoch+0)+".h5")

wH = wHistory()


batch_size = 32 # Batch size for training. //32 is the good
epochs =  100 # Number of epochs to train for.
latent_dim = 70 # Latent dimensionality of the encoding space.


import random
random.seed(seed)

jetNum=1000
valSplit=0.2

jetDim=200
trackNum =3# 10
genTrackNum=3

layNum = 4
parNum = 4

prob_thr =0.3

input_ = np.zeros(shape=(jetNum, jetDim,jetDim,layNum)) #jetMap
target_ = np.zeros(shape=(jetNum,jetDim, jetDim,trackNum,parNum))#+1
target_prob = np.zeros(shape=(jetNum,jetDim,jetDim,trackNum))

jetNum_test=50
input_test = np.zeros(shape=(jetNum_test, jetDim,jetDim,layNum)) #jetMap
target_test = np.zeros(shape=(jetNum_test,jetDim, jetDim,trackNum,parNum))#+1
target_prob_test = np.zeros(shape=(jetNum_test,jetDim,jetDim,trackNum))
input_jeta_test = np.zeros(shape=(jetNum_test))
input_jpt_test = np.zeros(shape=(jetNum_test))

openAngle=1#1 #NB never above pi/2!!!
layDist=3 #8 #3 is CMS
xyoff=1
bkgNum = 4
pixelInt = 100*1./100.

efficiency_4 = np.zeros(epochs)
fake_rate_4 = np.zeros(epochs)

efficiency_8 =  np.zeros(epochs)
fake_rate_8 = np.zeros(epochs)


# flag_jump = -9991

class validationCall(Callback) :
    def on_epoch_end(self,epoch, logs={}) :
        [call_par, call_prob] = self.model.predict(input_)

        for par in range(parNum) :
            bins = []# np.zeros(shape=(int(jetNum*valSplit)))
            nbin =0
            for j in range (int(jetNum*valSplit)) :
                j_eff = j+int(jetNum*(1-valSplit))
                for x in range(jetDim) :
                    for y in range(jetDim) :
                        for trk in range(trackNum) :
                            if call_prob[j_eff][x][y][trk] > prob_thr :
                                if call_par[j_eff][x][y][trk][0] != 0 or call_par[j_eff][x][y][trk][1] != 0  or call_par[j_eff][x][y][trk][2] != 0 or call_par[j_eff][x][y][3] != 0 :
                                    bins.append(call_par[j_eff][x][y][trk][par] - target_[j_eff][x][y][trk][par])
                                    nbin = nbin+1

            plt.figure()
            pylab.hist(bins,100, facecolor='green', alpha=0.75)
            pylab.title('parNum error distribution_ep{EPOCH}_par{PAR}'.format(PAR=par,EPOCH=epoch))
            pylab.ylabel('entry')
            pylab.xlabel('parNum error')
            plt.grid(True)
            # pylab.savefig("parameter_error_{EPOCH}_{PAR}.pdf".format(PAR=par,EPOCH=epoch))
            pdf_par.savefig()

        N_eff_4 = 0
        N_eff_8 = 0
        N_fake_4 =0
        N_fake_8 = 0
        N_tot_eff = jetNum*valSplit*genTrackNum
        N_tot_fake = 0
        for j in range (int(jetNum*valSplit)) :
            j_eff = j+int(jetNum*(1-valSplit))
            for x in range(jetDim) :
                for y in range(jetDim) :
                    for trk in range(trackNum) :
                        if target_prob[j_eff][x][y][trk]==1 :
                            chi2x = (call_par[j_eff][x][y][trk][0] - target_[j_eff][x][y][trk][0])**2
                            chi2y = (call_par[j_eff][x][y][trk][1] - target_[j_eff][x][y][trk][1])**2
                            chi2xt = (call_par[j_eff][x][y][trk][2] - target_[j_eff][x][y][trk][2])**2 / math.atan(2/float(layDist*3))
                            chi2yt = (call_par[j_eff][x][y][trk][3] - target_[j_eff][x][y][trk][3])**2 / math.atan(2/float(layDist*3))
                            chi2 = chi2x+chi2y+chi2xt+chi2yt
                            if chi2<=4  and call_prob[j_eff][x][y][trk]>prob_thr:
                                N_eff_4 = N_eff_4 +1
                            if chi2<=8  and call_prob[j_eff][x][y][trk]>prob_thr:
                                N_eff_8 = N_eff_8 +1
                        if call_prob[j_eff][x][y][trk] > prob_thr :
                            N_tot_fake = N_tot_fake +1
                            chi2x = (call_par[j_eff][x][y][trk][0] - target_[j_eff][x][y][trk][0])**2
                            chi2y = (call_par[j_eff][x][y][trk][1] - target_[j_eff][x][y][trk][1])**2
                            chi2xt = (call_par[j_eff][x][y][trk][2] - target_[j_eff][x][y][trk][2])**2 / math.atan(2/float(layDist*3))
                            chi2yt = (call_par[j_eff][x][y][trk][3] - target_[j_eff][x][y][trk][3])**2 / math.atan(2/float(layDist*3))
                            chi2 = chi2x+chi2y+chi2xt+chi2yt
                            if chi2>=4  and target_prob[j_eff][x][y][trk]==1:
                                print("fake 4!")
                                N_fake_4 = N_fake_4 +1
                            if chi2>=8  and target_prob[j_eff][x][y][trk]==1:
                                print("fake 8!")
                                N_fake_8 = N_fake_8 +1

        efficiency_4[epoch] = N_eff_4/N_tot_eff
        if N_tot_fake == 0 :
            fake_rate_4[epoch] = 1
        else :
           fake_rate_4[epoch] = N_fake_4/N_tot_fake

        efficiency_8[epoch] = N_eff_8/N_tot_eff
        if N_tot_fake == 0  :
            fake_rate_8[epoch] = 1
        else :
           fake_rate_8[epoch] = N_fake_8/N_tot_fake



# _EPSILON = K.epsilon()
# def loss_mse_sel(y_true, y_pred,y_weight):
#     y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
#     out = K.square(y_pred - y_true)*(y_weight)
#     return K.mean(out, axis=-1)
#altra idea: aggiungi vettore al target e poi lo splitti dentro la loss in 2 sotto vettori e fai l'mse di quelli.


#--------------------------------------------- INPUT from ROOT conversion-------------------------------#
gpu=True
if convert :

    import ROOT
    from root_numpy import *
    from root_numpy import testdata

    tfile = ROOT.TFile(input_name)
    tree = tfile.Get('demo/NNPixSeedInputTree')

    input_ = tree2array(tree, branches=['cluster_measured'])
    input_=rec2array(input_)
    input_jeta = tree2array(tree, branches=['jet_eta'])
    input_jeta =rec2array(input_jeta)
    input_jpt = tree2array(tree, branches=['jet_pt'])
    input_jpt =rec2array(input_jpt)
    target_ = tree2array(tree, branches=['trackPar'])
    target_=rec2array(target_)
    target_prob = tree2array(tree, branches=['trackProb'])
    target_prob =rec2array(target_prob)


    print("input shape=", input_.shape)
    print("input len=", len(input_))

    print("input eta shape=", input_jeta.shape)
    print("input eta len=", len(input_jeta))

    print("input pt shape=", input_jpt.shape)
    print("input pt len=", len(input_jpt))

    print("target par shape=", target_.shape)
    print("target par len=", len(target_))

    print("target prob shape=", target_prob.shape)
    print("target prob len=", len(target_prob))


    print("loading data: completed")

    if(gpu==False) :
        np.savez("NNPixSeed_event_{ev}".format(ev=jetNum), input_=input_, input_jeta=input_jeta, input_jpt=input_jpt, target_=target_, target_prob =target_prob)

    print("saving data: completed")






#---------------------------------------------numpy INPUT -------------------------------#
if convert==False and gpu==True:
    print("loading data: start")
    loadedfile = np.load(input_name)
    input_= loadedfile['input_']
    input_jeta= loadedfile['input_jeta']
    input_jpt= loadedfile['input_jpt']
    target_= loadedfile['target_']
    target_prob= loadedfile['target_prob']

    # for jet in range(jetNum) :
    #     for trk in range(trackNum) :
    #         target_[jet][trk][0]+=jetDim/2
    #         target_[jet][trk][1]+=jetDim/2


    print("loading data: completed")

    test_sample_creation = False
    if test_sample_creation == True:
        for jj in range (jetNum_test) :
            j = jj+(int(len(input_))-jetNum_test-5)
            input_jeta_test[jj] = input_jeta[j]
            input_jpt_test[jj] = input_jpt[j]
            for x in range(jetDim) :
                for y in range(jetDim) :
                    for par in range(parNum) :
                        input_test[jj][x][y][par] = input_[j][x][y][par]
                        for trk in range(trackNum) :
                            target_test[jj][x][y][trk][par] = target_[j][x][y][trk][par]
                            target_prob_test[jj][x][y][trk] = target_prob[j][x][y][trk]

        np.savez("NNPixSeed_event_{ev}_test".format(ev=jetNum_test), input_=input_test, input_jeta=input_jeta_test, input_jpt=input_jpt_test, target_=target_test, target_prob =target_prob_test)


#-----------------------------------------KERAS MODEL -----------------------------------#

if train or predict :

    NNinputs_jeta = Input(shape=(1,))
    NNinputs_jpt = Input(shape=(1,))
    NNinputsJet = concatenate([NNinputs_jeta,NNinputs_jpt])
    jetReshaped = Reshape((1,1,2))(NNinputsJet)
    jetUps = UpSampling2D(size=(jetDim,jetDim), data_format="channels_last")(jetReshaped)
    print("jetUps=", jetUps.shape)
    # print("jetUps size", jetUps.size)
    NNinputs = Input(shape=(jetDim,jetDim,layNum))
    print("NNinputs=", NNinputs.shape)
    # print("NNinputs size ", NNinputs.size)
    ComplInput = concatenate([NNinputs,jetUps],axis=3)
    print("ComplInput=", ComplInput.shape)
    # print("ComplInput size", ComplInput.size)

    # target_loss_w = np.zeros(shape=(jetNum,jetDim, jetDim,trackNum,parNum))#+1
    # for j in range (int(len(input_))) :
    #     for x in range(jetDim) :
    #         for y in range(jetDim) :
    #             for par in range(parNum) :
    #                 for trk in range(trackNum) :
    #                     if target_[j_eff][x][y][trk][0] = 0.0  and target_[j_eff][x][y][trk][1] = 0.0  and target_[j_eff][x][y][trk][2] = 0.0  and target_[j_eff][x][y][trk][3] = 0.0 :
    #                         target_w[j_eff][x][y][trk][par] = 0.0
    #                     else :
    #                         target_loss_w[j_eff][x][y][trk][par] = 1.0





    conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)

    conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2)
    conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3)
    reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)

    conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)

    # conv30_9 = Conv2D(100,31, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(50,21, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # conv30_5 = Conv2D(30,15, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    # conv20_5 = Conv2D(20,13, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(18,9, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    # # conv15_5 = Conv2D(18,9, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    #
    # conv15_3_1 = Conv2D(15,7, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # conv15_3_2 = Conv2D(15,5, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    # conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2)
    # conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    #
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)


    model = Model([NNinputs,NNinputs_jeta,NNinputs_jpt],[reshaped,reshaped_prob])

    #anubi = keras.optimizers.Adam(lr=0.001)

    #model.compile(optimizer=anubi, loss=['mse','binary_crossentropy'], loss_weights=[1,1]) #0.01,100
    model.compile(optimizer='adam', loss=['mse','binary_crossentropy'], loss_weights=[1,1]) #0.01,100


    model.summary()


# model = Sequential([
#         Dense(200, activation = 'relu', input_shape=(layNum,jetDim,jetDim) ),
#         Flatten(),
#         Dropout(0.2),
#         Dense(4)
#     ])

#-----------------------------------------NN TRAINING and PREDICITION -----------------------------------#
continue_training = True
combined_training = False

checkpointer = ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5",verbose=1, save_weights_only=True)
if train :
    if continue_training :
        model.load_weights('NNPixSeed_train_event_{ev}_bis.h5'.format(ev=jetNum))
        #model.load_weights('weights.16-0.00.hdf5')
        history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, nb_epoch=epochs+70, verbose = 2, validation_split=valSplit,  initial_epoch=70, callbacks=[checkpointer],class_weight={'reshaped':{},'reshaped_prob':{0:1,1:2000}})  #, callbacks=[validationCall()])
        model.save_weights('NNPixSeed_train_event_{ev}_tris.h5'.format(ev=jetNum))
    # elif combined_training :
    #     # model.load_weights('toyNN_train_bis_17_lay2_comp.h5')
    #     model.load_weights('toyNN_train_COMB_8_lay2_comp.h5')
    #     history  = model.fit(input_, [target_,target_prob],  batch_size=batch_size, nb_epoch=230+epochs, verbose = 2, validation_split=valSplit, initial_epoch=230+1,  callbacks=[validationCall()])
    #     model.save_weights('toyNN_train_COMB_{Seed}_bis_lay2_comp.h5'.format(Seed=seed))
    else :
        # pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_{Seed}_ep{Epoch}.pdf".format(Seed=seed, Epoch=epochs))
        history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit,  callbacks=[checkpointer],class_weight={'reshaped':{},'reshaped_prob':{0:1,1:2000}})  #, callbacks=[validationCall()])
        # pdf_par.close()
        model.save_weights('NNPixSeed_train_event_{ev}.h5'.format(ev=jetNum))


    pdf_loss = mpl.backends.backend_pdf.PdfPages("loss_file_ep{Epoch}_event{ev}.pdf".format( Epoch=epochs,ev=jetNum))

    plt.figure(1000)
    plt.yscale('log')
    pylab.plot(history.history['loss'])
    pylab.plot(history.history['val_loss'])
    pylab.title('model loss')
    pylab.ylabel('loss')
    pylab.xlabel('epoch')
    plt.grid(True)
    pylab.legend(['train', 'test'], loc='upper right')
    #pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
    pdf_loss.savefig(1000)
    # pylab.show()

    plt.figure(1001)
    plt.yscale('log')
    pylab.plot(history.history['reshape_2_loss'])
    pylab.plot(history.history['val_reshape_2_loss'])
    pylab.title('model loss (parameters)')
    pylab.ylabel('loss')
    pylab.xlabel('epoch')
    plt.grid(True)
    pylab.legend(['train', 'test'], loc='upper right')
    # pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
    pdf_loss.savefig(1001)

    plt.figure(1002)
    plt.yscale('log')
    pylab.plot(history.history['reshape_3_loss'])
    pylab.plot(history.history['val_reshape_3_loss'])
    pylab.title('model loss (probability)')
    pylab.ylabel('loss')
    pylab.xlabel('epoch')
    plt.grid(True)
    pylab.legend(['train', 'test'], loc='upper right')
    # pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
    pdf_loss.savefig(1002)

    # plt.figure(1003)
    # pylab.plot(efficiency_4)
    # pylab.plot(efficiency_8)
    # pylab.title('Efficiency of track finding')
    # pylab.ylabel('Efficiency')
    # pylab.xlabel('epoch')
    # plt.grid(True)
    # pylab.legend(['thr=4', 'thr=8'], loc='upper left')
    # pdf_loss.savefig(1003)
    #
    #
    # plt.figure(1004)
    # pylab.plot(fake_rate_4)
    # pylab.plot(fake_rate_8)
    # pylab.title('Fake Rate')
    # pylab.ylabel('Fake Rate')
    # pylab.xlabel('epoch')
    # plt.grid(True)
    # pylab.legend(['thr=4', 'thr=8'], loc='upper right')
    # pdf_loss.savefig(1004)

    pdf_loss.close()


if predict :

    print("prediction: start ")

    if train == False :
        model.load_weights('weights.115-0.00.hdf5')
        #model.load_weights('NNPixSeed_train_event_{ev}_bis.h5'.format(ev=jetNum))
        #model.load_weights('../new_deltaphi/NNPixSeed_train_event_1000.h5')

    [validation_par,validation_prob] = model.predict([input_,input_jeta,input_jpt])

    np.savez("NNPixSeed_prediction_event_{ev}".format(ev=jetNum), validation_par=validation_par, validation_prob=validation_prob)

    print("prediction: completed ")




#------------------------------------------------ PRINT ROOT FILE -----------------------------------#

if output :
     if predict == False :

        print("prediction loading: start")
        loadpred = np.load("NNPixSeed_prediction_event_{ev}.npz".format(ev=jetNum))

        validation_par = loadpred['validation_par']
        validation_prob = loadpred['validation_prob']

        print("prediction loading: completed")

     from ROOT import *
     gROOT.Reset()
     gROOT.SetBatch(True); #no draw at screen
     numPrint = 10
     validation_offset=int(len(input_)*(1-valSplit)+1)

     canvasTot = []
     canvasProb = []

     mapTot = []
     graphTargetTot = []
     mapProbPredTot = []
     graphPredTot = []


     for jet in range(numPrint) :

         canvasTot_jet = []
         mapTot_jet = []
         graphTargetTot_jet = []
         canvasProb_jet =[]
         mapProbPredTot_jet =[]
         graphPredTot_jet = []


         for trk in range(trackNum) :
            canvasProb_jet.append(TCanvas("canvasProb_%d_%d" % (jet,trk), "canvasProb_%d_%d" % (jet,trk), 600,800))
            mapProbPredTot_jet.append(TH2F("mapProbPredTot_%d_%d" % (jet,trk), "mapProbPredTot_%d_%d" % (jet,trk), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))

         for lay in range(layNum) :

             mapTot_jet.append(TH2F("mapTot_%d_%d" % (jet, lay), "mapTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             canvasTot_jet.append(TCanvas("canvasTot_%d_%d" % (jet, lay), "canvasTot_%d_%d" % (jet, lay), 600,800))
             graphTargetTot_jet.append(TGraph(genTrackNum))
            #  graphPredTot_jet.append(TGraph(trackNum*3))
             graphPredTot_jet.append(TGraph(jetDim*jetDim))


         mapTot.append(mapTot_jet)
         canvasTot.append(canvasTot_jet)
         graphTargetTot.append(graphTargetTot_jet)
         mapProbPredTot.append(mapProbPredTot_jet)
         canvasProb.append(canvasProb_jet)
         graphPredTot.append(graphPredTot_jet)





     for jet in range(numPrint) :
         print("-----------------------------------------------------------------------------")
         print("============================================================================================")
         print("-----------------------------------------------------------------------------")

         j_eff = jet+validation_offset
         j_eff = jet #ATTENZIONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         for lay in range(layNum) :
             tarPoint = 0
             predPoint = 0
             graphPredTot[jet][lay].SetMarkerColor(3)
             graphPredTot[jet][lay].SetMarkerStyle(28)
             graphPredTot[jet][lay].SetMarkerSize(3)
             graphTargetTot[jet][lay].SetMarkerColor(2)
             graphTargetTot[jet][lay].SetMarkerStyle(2)
             graphTargetTot[jet][lay].SetMarkerSize(3)
             for x in range(jetDim) :
                 for y in range(jetDim) :
                     mapTot[jet][lay].SetBinContent(x+1,y+1,input_[j_eff][x][y][lay])
                     for trk in range(trackNum) :
                             if(trk>0 and target_prob[j_eff][x][y][trk] == 1) :
                                print("Secondary map filled: map, x,y,jet",trk,x,y,jet)
                             mapProbPredTot[jet][trk].SetBinContent(x+1,y+1,validation_prob[j_eff][x][y][trk])

                             if target_prob[j_eff][x][y][trk] == 1 and lay==1:
                                 xx= target_[j_eff][x][y][trk][0]
                                 yy= target_[j_eff][x][y][trk][1]
                                 graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                 tarPoint = tarPoint+1
                             if validation_prob[j_eff][x][y][trk] > prob_thr and lay==1 :
                                 xx_pr= validation_par[j_eff][x][y][trk][0]
                                 yy_pr= validation_par[j_eff][x][y][trk][1]
                                 graphPredTot[jet][lay].SetPoint(predPoint,x+xx_pr-jetDim/2,y+yy_pr-jetDim/2)
                                 predPoint = predPoint+1
                                 print("________________________________________")
                                 print("New Pred, bin (x,y):",x,y)
                                 print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3], "Probabiity target=", target_prob[j_eff][x][y][trk])
                                 print("prediction(x,y,eta,phi)=",validation_par[j_eff][x][y][trk][0]," ", validation_par[j_eff][x][y][trk][1]," ",validation_par[j_eff][x][y][trk][2]," ",validation_par[j_eff][x][y][trk][3])
                            #  if(target_[j_eff][x][y][trk][0]!=0.0 or target_[j_eff][x][y][trk][1]!=0.0 or target_[j_eff][x][y][trk][2]!=0.0 or target_[j_eff][x][y][trk][3]!=0.0 ) :
                            #       print("---------------")
                            #       print("New Not-null-Target, bin (x,y):",x,y)
                            #       print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3], "Probabiity target=", target_prob[j_eff][x][y][trk])
                            #       print("prediction(x,y,eta,phi)=",validation_par[j_eff][x][y][trk][0]," ", validation_par[j_eff][x][y][trk][1]," ",validation_par[j_eff][x][y][trk][2]," ",validation_par[j_eff][x][y][trk][3])

     output_file = TFile("NNPixSeed_mapValidation_events_{ev}.root".format(ev=jetNum),"recreate")

     for jet in range(numPrint) :
         for lay in range(layNum) :
             canvasTot[jet][lay].cd()
             mapTot[jet][lay].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].GetYaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].Draw("colz")
             graphTargetTot[jet][lay].Draw("SAME P")
             graphPredTot[jet][lay].Draw("SAME P")
             canvasTot[jet][lay].Write()

         for trk in range(trackNum) :
             canvasProb[jet][trk].cd()
             mapProbPredTot[jet][trk].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapProbPredTot[jet][trk].GetYaxis().SetRangeUser(-jetDim,jetDim)
             mapProbPredTot[jet][trk].Draw("colz")
             graphTargetTot[jet][1].Draw("SAME P")
             graphPredTot[jet][1].Draw("SAME P")
             canvasProb[jet][trk].Write()

     output_file.Close()


     print("parameter file: start looping")
     pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_events_{ev}.pdf".format(ev=jetNum))
     for par in range(parNum) :
         bins = []# np.zeros(shape=(int(jetNum*valSplit)))
         bins_pred = []
         bins_target = []
         nbin =0
         for j in range (int(len(input_)*valSplit)) :
             j_eff = j+int(len(input_)*(1-valSplit))
             for x in range(jetDim) :
                 for y in range(jetDim) :
                     for trk in range(trackNum) :
                         if validation_prob[j_eff][x][y][trk] > prob_thr :
                             if validation_par[j_eff][x][y][trk][0] != 0 or validation_par[j_eff][x][y][trk][1] != 0  or validation_par[j_eff][x][y][trk][2] != 0 or validation_par[j_eff][x][y][3] != 0 :
                                 bins.append(validation_par[j_eff][x][y][trk][par] - target_[j_eff][x][y][trk][par])
                                 bins_pred.append(validation_par[j_eff][x][y][trk][par])
                                 bins_target.append(target_[j_eff][x][y][trk][par])
                                 nbin = nbin+1

         plt.figure()
         plt.yscale('log')
         pylab.hist(bins,200, facecolor='green', alpha=0.75)
         pylab.title('Residual distribution_par{PAR}'.format(PAR=par))
         pylab.ylabel('entry')
         pylab.xlabel('prediction-target')
         plt.grid(True)
         # pylab.savefig("parameter_error_{EPOCH}_{PAR}.pdf".format(PAR=par,EPOCH=epoch))
         pdf_par.savefig()

         plt.figure()
         plt.yscale('log')
         pylab.hist(bins_target,200, facecolor='red', alpha=0.75)
         pylab.title('Target distribution_par{PAR}'.format(PAR=par))
         pylab.ylabel('entry')
         pylab.xlabel('target')
         plt.grid(True)
         pdf_par.savefig()

         plt.figure()
         plt.yscale('log')
         pylab.hist(bins_pred,200, facecolor='blue', alpha=0.75)
         pylab.title('Prediction distribution_par{PAR}'.format(PAR=par))
         pylab.ylabel('entry')
         pylab.xlabel('prediction')
         plt.grid(True)
         pdf_par.savefig()

     pdf_par.close()
