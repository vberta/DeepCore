from __future__ import print_function
import os
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['GOTO_NUM_THREADS'] = '40'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['openmp'] = 'True'

from keras.callbacks import Callback
from keras.models import Model,load_model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Conv2DTranspose, concatenate, Concatenate, ZeroPadding2D
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

from  matplotlib import pyplot as plt
import pylab
import glob

#######################################
#
# USAGE: python yoloJet.py --seed SEED --simulator --input INPUT --training --predict --output
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
    '--simulator',
    dest='Simulator',
    action='store_const',
    const=True,
    default=False,
    help='activate the toy MC production')
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
generate = args.Simulator
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


batch_size = 128 # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 70 # Latent dimensionality of the encoding space.


import random
random.seed(seed)

jetNum=500000
valSplit=0.2


jetDim=30
trackNum =3# 10
layNum = 4
parNum = 4

input_ = np.zeros(shape=(jetNum, jetDim,jetDim,layNum)) #jetMap
target_ = np.zeros(shape=(jetNum,jetDim, jetDim,trackNum,parNum+1))

openAngle=1#1 #NB never over pi/2!!!
layDist=3 #8 #3 is CMS
xyoff=1
bkgNum = 4
pixelInt = 100*1./100.

class validationCall(Callback) :
    def on_epoch_end(self,epoch, logs={}) :
        call_par = self.model.predict(input_)

        for par in range(parNum) :
            bins = np.zeros(shape=(int(jetNum*valSplit)))
            for j in range (int(jetNum*valSplit)) :
                j_eff = j+int(jetNum*(1-valSplit))
                bins[j] = call_par[j_eff][par] - target_par[j_eff][par]
            plt.figure()
            pylab.hist(bins,100, facecolor='green', alpha=0.75)
            pylab.title('parNum error distribution_ep{EPOCH}_par{PAR}'.format(PAR=par,EPOCH=epoch))
            pylab.ylabel('entry')
            pylab.xlabel('parNum error')
            plt.grid(True)
            # pylab.savefig("parameter_error_{EPOCH}_{PAR}.pdf".format(PAR=par,EPOCH=epoch))
            pdf_par.savefig()


#------------------------------------------------ TOY MC DEFINITION ------------------------------------------------#

if generate :
    for i in range(jetNum) :
        print("-------------------------- ")
        print("sample ",i)
        jetMap= np.zeros(shape=(jetDim,jetDim,layNum))
        jetPar = np.zeros(shape=(jetDim, jetDim,trackNum,parNum+1))

        trackPar= np.zeros(shape=(trackNum,parNum))
        xy_add= np.zeros(shape=(trackNum,parNum-2))

        jetDirR=random.uniform(0,openAngle)
        jetDirPhi=random.uniform(0,2*math.pi)

        #nTracks=random.randint(0,trackNum)
        nTracks=3
        distThr = 4 #threshold in maximum distance between pixel and track recorded in the pixel

        trackDirR = np.zeros(trackNum)
        trackDirPhi = np.zeros(trackNum)
        trackX = np.zeros(trackNum)
        trackY = np.zeros(trackNum)

        for trk in range(nTracks) :

           locR = random.uniform(0,openAngle)
           locPhi = random.uniform(0,2*math.pi)
           x_over_d=math.tan(locR)*math.sin(locPhi)+math.tan(jetDirR)*math.sin(jetDirPhi)
           y_over_d=math.tan(locR)*math.cos(locPhi)+math.tan(jetDirR)*math.cos(jetDirPhi)
           tanR=math.sqrt(math.pow(x_over_d,2)+math.pow(y_over_d,2))

           if(tanR ==0.0) :
               tanR = 0.00001

           trackDirR[trk]=math.atan(tanR)
           trackDirPhi[trk] = math.asin(x_over_d/tanR)+math.pi

           trackX[trk]=random.uniform(0,xyoff)
           trackY[trk]=random.uniform(0,xyoff)

        for lay in range(layNum) :
           for trk in range(nTracks) :
               d = (lay+1)*layDist*math.tan(trackDirR[trk])
               x = d*(math.sin(trackDirPhi[trk])+trackX[trk])
               y = d*(math.cos(trackDirPhi[trk])+trackY[trk])
               for electrons in range(100) :
                       channelx=int(random.gauss(x,0.3))
                       channely=int(random.gauss(y,0.3))
                       if channelx>=-jetDim/2 and channelx < jetDim/2 and channely>=-jetDim/2 and channely < jetDim/2 :
                           jetMap[channelx+jetDim/2][channely+jetDim/2][lay]+=pixelInt
               if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2  and lay==0:
                   xy_add[trk] = (x,y)
               if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2  and lay==1:
                    # thetaX = math.atan(math.tan(trackDirR[trk])*math.sin(trackDirPhi[trk]))
                    # thetaY = math.atan(math.tan(trackDirR[trk])*math.cos(trackDirPhi[trk]))
                    thetaX = math.atan((x-xy_add[trk][0])/layDist)
                    thetaY = math.atan((y-xy_add[trk][1])/layDist)
                    trackPar[trk] = (x+jetDim/2,y+jetDim/2,thetaX,thetaY)
                    # print("generated (x,y)= ", x+jetDim/2, y+jetDim/2)
               elif lay == 1:
                   print("out of window range:",x,y, lay, trk)
                #    trackPar[trk] = (jetDim/2,jetDim/2,0,0)
                   trackPar[trk] = (-999,-999,-999,-999)

        for x in range(jetDim) :
            for y in range(jetDim) :
                pixObj = []
                pixPar = np.zeros(shape=(trackNum,parNum+1))
                dist4sort = np.zeros(shape=(trackNum))
                for trk in range(nTracks) :
                    if int(trackPar[trk][0]) == x and int(trackPar[trk][1]) == y :
                        pixPar[trk][4] = 1
                    else :
                        pixPar[trk][4] = 0
                    xp = x + 0.5
                    yp = y + 0.5
                    distX =  trackPar[trk][0] -xp
                    distY = trackPar[trk][1] - yp
                    dist4sort[trk] = math.sqrt(distX**2+distY**2)

                    if abs(distX)<=distThr and abs(distY)<=distThr :
                        pixPar[trk][0] = distX
                        pixPar[trk][1] = distY
                        pixPar[trk][2] = trackPar[trk][2]
                        pixPar[trk][3] = trackPar[trk][3]
                        # if(jetPar[x][y][trk][4] ==1) :
                        #     print("recorded (x,y)",distX,distY)
                    else :
                        for p in range(parNum) :
                            pixPar[trk][p] = 0.0
                    if(trackPar[trk][0] == -999 and trackPar[trk][1] == -999 and trackPar[trk][2] == -999 and trackPar[trk][3] == -999) :
                    # if(trackPar[trk][0] == 0. and trackPar[trk][1] == 0. and trackPar[trk][2] == 0. and trackPar[trk][3] == 0.) :
                        # jetPar[x][y][trk][4] = 0
                        for p in range(parNum+1) :
                            pixPar[trk][4] = 0.0
                    pixObj.append((pixPar[trk],dist4sort[trk]))
                # if dist4sort[0] <1 or dist4sort[1] <1 or dist4sort[2] <1 :
                    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX=",x)
                    # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY=",y)
                    # print("PIXOBJ",pixObj)
                pixObj = sorted(pixObj, key= (lambda( pixObj) : pixObj[1]), reverse=False)
                pixParSorted = np.zeros(shape=(trackNum,parNum+1))
                dist4sortSorted = np.zeros(shape=(trackNum))
                for trk in range(trackNum) :
                    (pixParSorted[trk],dist4sortSorted[trk]) = pixObj[trk]
                # if dist4sort[0] <1 or dist4sort[1] <1 or dist4sort[2] <1 :
                    # print("SORTED",pixParSorted)
                jetPar[x][y] = pixParSorted

        input_[i] = jetMap
        target_[i] = jetPar

    np.savez("yoloJet_MC_event_{ev}_layer{llay}_angle{angle}_{seed}_new_lay2".format(ev=jetNum,llay=layNum, angle=openAngle, seed=seed), input_=input_, target_=target_)




#---------------------------------------------EXTERNAL INPUT -------------------------------#

if generate==False :
    loadedfile = np.load(input_name)
    input_= loadedfile['input_']
    target_= loadedfile['target_']

    # for jet in range(jetNum) :
    #     for trk in range(trackNum) :
    #         target_[jet][trk][0]+=jetDim/2
    #         target_[jet][trk][1]+=jetDim/2


    print("loaded complete")
#-----------------------------------------KERAS MODEL -----------------------------------#

if train or predict :

    # NNinputs = Input(shape=(jetDim,jetDim,layNum))
    # conv30_9 = Conv2D(30,9, data_format="channels_last", input_shape=(jetDim,jetDim,layNum), activation='relu',padding="same")(NNinputs)
    # conv30_7 = Conv2D(30,7, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # conv30_5 = Conv2D(30,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    # conv20_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(15,5, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    # conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_5)
    # reshaped = Reshape((30,30,trackNum,parNum+1))(conv15_3)

    NNinputs = Input(shape=(jetDim,jetDim,layNum))
    conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum), activation='relu',padding="same")(NNinputs)
    conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_5)
    reshaped = Reshape((30,30,trackNum,parNum+1))(conv15_3)

    model = Model(NNinputs,reshaped)

    model.compile('adam', 'mse')

    model.summary()


# model = Sequential([
#         Dense(200, activation = 'relu', input_shape=(layNum,jetDim,jetDim) ),
#         Flatten(),
#         Dropout(0.2),
#         Dense(4)
#     ])

#-----------------------------------------NN TRAINING and PREDICITION -----------------------------------#
continue_training = True
if train :
    if continue_training :
        model.load_weights('toyNN_train_{Seed}_lay2.h5'.format(Seed=seed))
        history  = model.fit(input_, target_,  batch_size=batch_size, nb_epoch=epochs+epochs, verbose = 2, validation_split=valSplit, initial_epoch=epochs+1) #TODO map pred
        model.save_weights('toyNN_train_bis_{Seed}.h5'.format(Seed=seed))
    else :
        pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_{Seed}_ep{Epoch}.pdf".format(Seed=seed, Epoch=epochs))
        history  = model.fit(input_, target_,  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit)#, callbacks=[validationCall()])
        pdf_par.close()
        model.save_weights('toyNN_train_{Seed}_lay2.h5'.format(Seed=seed))

    pdf_loss = mpl.backends.backend_pdf.PdfPages("loss_file_{Seed}_ep{Epoch}.pdf".format(Seed=seed, Epoch=epochs))
    plt.figure(1000)
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
    pdf_loss.close()


if predict :

    if train == False :
        model.load_weights('toyNN_train_{Seed}_lay2.h5'.format(Seed=seed))

    validation_par = model.predict(input_)

    np.savez("toyNN_prediction_{Seed}".format(Seed=seed), validation_par=validation_par)





#------------------------------------------------ PRINT ROOT FILE -----------------------------------#

if output :
     if predict == False :
        loadpred = np.load("toyNN_prediction_{Seed}.npz".format(Seed=seed))

        validation_par = loadpred['validation_par']


     from ROOT import *
     gROOT.Reset()
     gROOT.SetBatch(True); #no draw at screen
     numPrint = 20
     validation_offset=int(jetNum*(1-valSplit)+1)


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
             graphTargetTot_jet.append(TGraph(trackNum))
             graphPredTot_jet.append(TGraph(trackNum*3))

         mapTot.append(mapTot_jet)
         canvasTot.append(canvasTot_jet)
         graphTargetTot.append(graphTargetTot_jet)
         mapProbPredTot.append(mapProbPredTot_jet)
         canvasProb.append(canvasProb_jet)
         graphPredTot.append(graphPredTot_jet)





     for jet in range(numPrint) :
         print("--------------------------------")
         j_eff = jet+validation_offset
         #j_eff = jet
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

                        #  if lay==3 :
                             mapProbPredTot[jet][trk].SetBinContent(x+1,y+1,validation_par[j_eff][x][y][trk][4])
                             if validation_par[j_eff][x][y][trk][4]>0.9 :
                                 print("prediction>0.5 (x,y)=",x,y)
                             if target_[j_eff][x][y][trk][4] == 1 :
                                 xx= target_[j_eff][x][y][trk][0]-layDist*(1-lay)*math.tan(target_[j_eff][x][y][trk][2])
                                 yy= target_[j_eff][x][y][trk][1]-layDist*(1-lay)*math.tan(target_[j_eff][x][y][trk][3])
                                 print("TARGET_{TRK}_{LAY}_(x,y)=".format(TRK=tarPoint, LAY=lay),x,y,xx,yy)
                                 graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                 tarPoint = tarPoint+1
                             if validation_par[j_eff][x][y][trk][4] > 0.05 :
                                xx= validation_par[j_eff][x][y][trk][0]-layDist*(1-lay)*math.tan(validation_par[j_eff][x][y][trk][2])
                                yy= validation_par[j_eff][x][y][trk][1]-layDist*(1-lay)*math.tan(validation_par[j_eff][x][y][trk][3])
                                print("PREDICTION_{TRK}_{LAY}_(x,y)=".format(TRK=predPoint, LAY=lay),x,y,xx,yy)
                                # print("difference theta X=",validation_par[j_eff][x][y][trk][2]-target_[j_eff][x][y][trk][2])
                                # print("difference theta Y=",validation_par[j_eff][x][y][trk][3]-target_[j_eff][x][y][trk][3])
                                graphPredTot[jet][lay].SetPoint(predPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                print("pred point =",predPoint, lay, trk,)

                                predPoint = predPoint+1


     output_file = TFile("toyNN_{Seed}.root".format(Seed=seed),"recreate")

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
