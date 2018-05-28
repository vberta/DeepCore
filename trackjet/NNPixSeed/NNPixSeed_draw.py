from __future__ import print_function
import os
import numpy as np
import math
import sys
import argparse
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.backends.backend_pdf as backpdf

from  matplotlib import pyplot as plt
import pylab
import glob
import random

import ROOT
from root_numpy import *
from root_numpy import testdata


jetNum=50
valSplit=0.2

jetDim=200
trackNum =3# 10
genTrackNum=3

layNum = 4
parNum = 4

prob_thr =0.4

# input_ = np.zeros(shape=(jetNum, jetDim,jetDim,layNum)) #jetMap
# target_ = np.zeros(shape=(jetNum,jetDim, jetDim,trackNum,parNum))#+1
# target_prob = np.zeros(shape=(jetNum,jetDim,jetDim,trackNum))

openAngle=1#1 #NB never above pi/2!!!
layDist=3 #8 #3 is CMS
xyoff=1
bkgNum = 4
pixelInt = 100*1./100.


# filename = testdata.get_filepath('histo.root:/demo')
tfile = ROOT.TFile("histo.root")
tree = tfile.Get('demo/NNPixSeedInputTree')

input_ = tree2array(tree, branches=['cluster_measured'])
input_=rec2array(input_)
target_ = tree2array(tree, branches=['trackPar'])
target_=rec2array(target_)
target_prob = tree2array(tree, branches=['trackProb'])
target_prob =rec2array(target_prob)
# trakk = tree2array(tree, branches=['track_pt'])
# trakk=rec2array(trakk)


print(input_.shape)
print(target_.shape)
print(target_prob.shape)
# print(trakk.shape)

print("loading data: completed")
#-----------------------------------------KERAS MODEL -----------------------------------#

# if train or predict :
#     NNinputs = Input(shape=(jetDim,jetDim,layNum))
#     conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum), activation='relu',padding="same")(NNinputs)
#     conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
#     conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
#     conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
#     conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)
#
#     conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
#     conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
#     conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2)
#     conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3)
#     reshaped = Reshape((30,30,trackNum,parNum))(conv15_3)
#
#     conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
#     reshaped_prob = Reshape((30,30,trackNum))(conv1_3_1)
#
#     model = Model(NNinputs,[reshaped,reshaped_prob])
#     model.compile(optimizer='adam', loss=['mse','binary_crossentropy'], loss_weights=[1,1]) #0.01,100
#     model.summary()


#-----------------------------------------NN TRAINING and PREDICITION -----------------------------------#
# continue_training = False
# combined_training = False
# if train :
#     if continue_training :
#         model.load_weights('toyNN_train_{Seed}_lay2_comp.h5'.format(Seed=seed))
#         history  = model.fit(input_, [target_,target_prob],  batch_size=batch_size, nb_epoch=epochs+epochs, verbose = 2, validation_split=valSplit, initial_epoch=epochs+1,  callbacks=[validationCall()])
#         model.save_weights('toyNN_train_bis_{Seed}_lay2_comp.h5'.format(Seed=seed))
#     elif combined_training :
#         # model.load_weights('toyNN_train_bis_17_lay2_comp.h5')
#         model.load_weights('toyNN_train_COMB_8_lay2_comp.h5')
#         history  = model.fit(input_, [target_,target_prob],  batch_size=batch_size, nb_epoch=230+epochs, verbose = 2, validation_split=valSplit, initial_epoch=230+1,  callbacks=[validationCall()])
#         model.save_weights('toyNN_train_COMB_{Seed}_bis_lay2_comp.h5'.format(Seed=seed))
#     else :
#         pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_{Seed}_ep{Epoch}.pdf".format(Seed=seed, Epoch=epochs))
#         history  = model.fit(input_, [target_,target_prob],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit,  callbacks=[validationCall()])#, callbacks=[validationCall()])
#         pdf_par.close()
#         model.save_weights('toyNN_train_{Seed}_lay2_comp.h5'.format(Seed=seed))

    # pdf_loss = mpl.backends.backend_pdf.PdfPages("loss_file_{Seed}_ep{Epoch}_comp.pdf".format(Seed=seed, Epoch=epochs))
    #
    # plt.figure(1000)
    # pylab.plot(history.history['loss'])
    # pylab.plot(history.history['val_loss'])
    # pylab.title('model loss')
    # pylab.ylabel('loss')
    # pylab.xlabel('epoch')
    # plt.grid(True)
    # pylab.legend(['train', 'test'], loc='upper right')
    # pdf_loss.savefig(1000)
    #
    # plt.figure(1001)
    # pylab.plot(history.history['reshape_1_loss'])
    # pylab.plot(history.history['val_reshape_1_loss'])
    # pylab.title('model loss (parameters)')
    # pylab.ylabel('loss')
    # pylab.xlabel('epoch')
    # plt.grid(True)
    # pylab.legend(['train', 'test'], loc='upper right')
    # pdf_loss.savefig(1001)
    #
    # plt.figure(1002)
    # pylab.plot(history.history['reshape_2_loss'])
    # pylab.plot(history.history['val_reshape_2_loss'])
    # pylab.title('model loss (probability)')
    # pylab.ylabel('loss')
    # pylab.xlabel('epoch')
    # plt.grid(True)
    # pylab.legend(['train', 'test'], loc='upper right')
    # pdf_loss.savefig(1002)
    #
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
    #
    # pdf_loss.close()


# if predict :
#
#     if train == False :
#         # model.load_weights('toyNN_train_{Seed}_lay2_comp.h5'.format(Seed=seed)
#         #model.load_weights('toyNN_train_COMB_8_lay2_comp.h5')
#
#         # model.load_weights('toyNN_train_COMB_8_bis_lay2_comp.h5')
#         model.load_weights('toyNN_train_15_lay2_comp.h5')
#
#     [validation_par,validation_prob] = model.predict(input_)
#     np.savez("toyNN_prediction_{Seed}_lay2_comp".format(Seed=seed), validation_par=validation_par, validation_prob=validation_prob)





#------------------------------------------------ PRINT ROOT FILE -----------------------------------#
output=True
predict = False
if output :
    #  if predict == False :
     #
    #     print("prediction loading: start")
    #     # loadpred = np.load("toyNN_prediction_{Seed}_lay2_comp.npz".format(Seed=seed))
     #
    #     # validation_par = loadpred['validation_par']
    #     # validation_prob = loadpred['validation_prob']
     #
    #     print("prediction loading: completed")

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
        #  mapProbPredTot_jet =[]
        #  graphPredTot_jet = []


         for trk in range(trackNum) :
            canvasProb_jet.append(TCanvas("canvasProb_%d_%d" % (jet,trk), "canvasProb_%d_%d" % (jet,trk), 600,800))
            # mapProbPredTot_jet.append(TH2F("mapProbPredTot_%d_%d" % (jet,trk), "mapProbPredTot_%d_%d" % (jet,trk), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))

         for lay in range(layNum) :

             mapTot_jet.append(TH2F("mapTot_%d_%d" % (jet, lay), "mapTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             canvasTot_jet.append(TCanvas("canvasTot_%d_%d" % (jet, lay), "canvasTot_%d_%d" % (jet, lay), 600,800))
             graphTargetTot_jet.append(TGraph(genTrackNum))
            #  graphPredTot_jet.append(TGraph(jetDim*jetDim))


         mapTot.append(mapTot_jet)
         canvasTot.append(canvasTot_jet)
         graphTargetTot.append(graphTargetTot_jet)
        #  mapProbPredTot.append(mapProbPredTot_jet)
         canvasProb.append(canvasProb_jet)
        #  graphPredTot.append(graphPredTot_jet)





     for jet in range(numPrint) :
         print("--------------------------------")
         j_eff = jet+validation_offset
         j_eff = jet #ATTENZIONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         for lay in range(layNum) :
             tarPoint = 0
             predPoint = 0
            #  graphPredTot[jet][lay].SetMarkerColor(3)
            #  graphPredTot[jet][lay].SetMarkerStyle(28)
            #  graphPredTot[jet][lay].SetMarkerSize(3)
             graphTargetTot[jet][lay].SetMarkerColor(2)
             graphTargetTot[jet][lay].SetMarkerStyle(2)
             graphTargetTot[jet][lay].SetMarkerSize(3)
             for x in range(jetDim) :
                 for y in range(jetDim) :
                     mapTot[jet][lay].SetBinContent(x+1,y+1,input_[j_eff][x][y][lay])
                     if(lay==1 and input_[j_eff][x][y][lay]!=0 ) :
                         print("map, x,y=", x+1-jetDim/2,y+1-jetDim/2, "(", x,y,")")
                    #  if(input_[j_eff][x][y][lay]!=0) :
                    #      print("input,x,y,lay, val=",x,y,lay, input_[j_eff][x][y][lay])
                     for trk in range(trackNum) :
                            #  if(trk>0 and target_prob[j_eff][x][y][trk] == 1) :
                            #     print("Secondary map filled: map, x,y,jet",trk,x,y,jet)
                            #  mapProbPredTot[jet][trk].SetBinContent(x+1,y+1,validation_prob[j_eff][x][y][trk])
                            #  if(jet==1 and trk ==1) :
                            #      print("x,y,trk,prob",x,y,trk,validation_prob[j_eff][x][y][trk])
                            #  mapProbPredTot[jet][trk].SetBinContent(x+1,y+1,validation_par[j_eff][x][y][trk][4])

                            #  if validation_prob[j_eff][x][y][trk]>0.9 :
                            #       print("prediction>0.9 (x,y)=",x,y)
                            #  if target_[j_eff][x][y][trk][4] == 1 :
                             if target_prob[j_eff][x][y][trk] == 1 :
                                #  xx= target_[j_eff][x][y][trk][0]-layDist*(1-lay)*math.tan(target_[j_eff][x][y][trk][2])
                                #  yy= target_[j_eff][x][y][trk][1]-layDist*(1-lay)*math.tan(target_[j_eff][x][y][trk][3])
                                 if(lay==1) :
                                    print("prob1, x,y=", x-jetDim/2,y-jetDim/2, "(", x,y,")")
                                    xx = target_[j_eff][x][y][trk][0]
                                    yy = target_[j_eff][x][y][trk][1]
                                #  print("TARGET_{TRK}_{LAY}_(x,y)=".format(TRK=tarPoint, LAY=lay),x,y,xx,yy)
                                    graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                    tarPoint = tarPoint+1

                                #  if(lay==1) : print("filled prob!!")
                            #  if validation_par[j_eff][x][y][trk][4] > 0.05 :
                            #  if validation_prob[j_eff][x][y][trk] > prob_thr :
                            #      xx_pr= validation_par[j_eff][x][y][trk][0]-layDist*(1-lay)*math.tan(validation_par[j_eff][x][y][trk][2])
                            #      yy_pr= validation_par[j_eff][x][y][trk][1]-layDist*(1-lay)*math.tan(validation_par[j_eff][x][y][trk][3])
                            #      # print("PREDICTION_{TRK}_{LAY}_(x,y)=".format(TRK=predPoint, LAY=lay),x,y,xx,yy)
                            #      # print("difference theta X=",validation_par[j_eff][x][y][trk][2]-target_[j_eff][x][y][trk][2])
                            #      # print("difference theta Y=",validation_par[j_eff][x][y][trk][3]-target_[j_eff][x][y][trk][3])
                            #      graphPredTot[jet][lay].SetPoint(predPoint,x+xx_pr-jetDim/2,y+yy_pr-jetDim/2)
                            #      # print("pred point =",predPoint, lay, trk,)
                            #      predPoint = predPoint+1


     output_file = TFile("toyNN_{Seed}.root".format(Seed="prova"),"recreate")

     for jet in range(numPrint) :
         for lay in range(layNum) :
             canvasTot[jet][lay].cd()
             mapTot[jet][lay].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].GetYaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].Draw("colz")
             graphTargetTot[jet][lay].Draw("SAME P")
            #  graphPredTot[jet][lay].Draw("SAME P")
             canvasTot[jet][lay].Write()

         for trk in range(1) : #trackNum
             canvasProb[jet][trk].cd()
            #  mapProbPredTot[jet][trk].GetXaxis().SetRangeUser(-jetDim,jetDim)
            #  mapProbPredTot[jet][trk].GetYaxis().SetRangeUser(-jetDim,jetDim)
            #  mapProbPredTot[jet][trk].Draw("colz")
            #  graphTargetTot[jet][1].Draw(" P")
            #  graphPredTot[jet][1].Draw("SAME P")
            #  canvasProb[jet][trk].Write()
             graphTargetTot[jet][1].GetXaxis().SetRangeUser(-jetDim/2,jetDim/2)
             graphTargetTot[jet][1].GetYaxis().SetRangeUser(-jetDim/2,jetDim/2)
            #  graphTargetTot[jet][1].Write("P")

     output_file.Close()
