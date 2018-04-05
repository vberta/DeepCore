from __future__ import print_function
import os
os.environ['MKL_NUM_THREADS'] = '40'
os.environ['GOTO_NUM_THREADS'] = '40'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['openmp'] = 'True'

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

from  matplotlib import pyplot as plt
import pylab
import glob

#######################################
#
# USAGE: python toyNN.py --seed SEED --simulator --input INPUT --training --predict --output
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

one_track = True
one_track_only = True


with tf.Session(config=tf.ConfigProto(
                   intra_op_parallelism_threads=40)) as sess:
   K.set_session(sess)

class wHistory(keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs={}):
       if epoch % 10 == 0 :
               self.model.save("trained"+str(epoch+0)+".h5")

wH = wHistory()


batch_size = 128 # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 70 # Latent dimensionality of the encoding space.


import random
#random.seed(8) #l'ha detto tommaso
random.seed(seed)

jetNum=500000#100000
valSplit=0.2
# slen=30
# fraction=0.15 #????
# inputdata=[]

jetDim=30
trackNum =3# 10
layNum = 4
parNum = 4

input_ = np.zeros(shape=(jetNum,layNum, jetDim,jetDim)) #jetMap
if one_track_only :
    target_par = np.zeros(shape=(jetNum, parNum))
    target_map = np.zeros(shape=(jetNum, layNum, jetDim,jetDim))
elif generate :
    target_par = np.zeros(shape=(jetNum, trackNum, parNum)) #4=x,y,eta,phi#TODO 10par +2*layNum
    target_par0 = np.zeros(shape=(jetNum, parNum)) #4=x,y,eta,phi#TODO 10par
    target_map = np.zeros(shape=(jetNum, trackNum, layNum, jetDim,jetDim)) #trackMap
    target_map0 = np.zeros(shape=(jetNum, layNum, jetDim,jetDim)) #trackMap
elif one_track :
    target_par = np.zeros(shape=(jetNum, parNum))
    target_map = np.zeros(shape=(jetNum, layNum, jetDim,jetDim))
else :
    target_par = np.zeros(shape=(jetNum, trackNum, parNum))
    target_map = np.zeros(shape=(jetNum, trackNum, layNum, jetDim,jetDim))



# validation_input_ = np.zeros(shape=(jetNum,layNum, jetDim,jetDim)) #jetMap
# validation_target_par = np.zeros(shape=(jetNum, parNum)) #4=x,y,eta,phi         #FIXME one dimension (trk) missing for toy simplification
# validation_target_map = np.zeros(shape=(jetNum, trackNum, layNum, jetDim,jetDim)) #trackMap

openAngle=1 #NB never over pi/2!!!
layDist=3 #8 #3 is CMS
xyoff=1


def mindist(trackObj, trk) :
   maps = [x[1] for x in trackObj]
   #print(maps)
   # minimum= min([(maps[t][0]-ix)**2+(maps[t][1]-iy)**2 for (ix,iy) in maps[t][x,y] if ix != maps[0] and iy != maps[1]] or [0] for t in )
   # minimum= min([(maps[t][0]-ix)**2+(maps[1]-iy)**2 for ix in maps[]])
   #minimum = maps[0]
   dist = []
   for t in range(trackNum):
       if(t!=trk) :
            dist.append((maps[trk][1]-maps[t][1])**2+(maps[trk][2]-maps[t][2])**2)
   min_dist = min(dist)
   return min_dist

def mindistMod(maps, trk) : #TODO new implementation TODO 4 par
   dist = []
   for t in range(trackNum):
       if(t!=trk) :
            dist.append((maps[trk][0]-maps[t][0])**2+(maps[trk][1]-maps[t][1])**2)
   min_dist = min(dist)
   return min_dist

# def mindistMod(maps, trk) : #TODO new implementation TODO 10 par
#    dist = []
#    d0 = layDist*math.tan(maps[trk][2])
#    x0 = d0*(math.sin(maps[trk][3])+maps[trk][0])
#    y0 = d0*(math.cos(maps[trk][3])+maps[trk][1])
#    for t in range(trackNum):
#        if(t!=trk) :
#             dd = layDist*math.tan(maps[t][2])
#             xx = dd*(math.sin(maps[t][3])+maps[t][0])
#             yy = dd*(math.cos(maps[t][3])+maps[t][1])
#             dist.append((x0-xx)**2+(y0-yy)**2)
#    min_dist = min(dist)
#    return min_dist



#------------------------------------------------ TOY MC DEFINITION ------------------------------------------------#

if generate :
    for i in range(jetNum) :

        print("sample ",i)
        jetMap= np.zeros(shape=(layNum, jetDim,jetDim))
        trackMap= np.zeros(shape=(trackNum,layNum, jetDim,jetDim))
        trackPar= np.zeros(shape=(trackNum,parNum)) #parNum #TODO 4 par
        # trackPar= np.zeros(shape=(trackNum,parNum+2*layNum)) #parNum #TODO 10par


        trackObj = [] #objet useful for sorting only

        jetDirR=random.uniform(0,openAngle)
        jetDirPhi=random.uniform(0,2*math.pi)

        #nTracks=random.randint(0,trackNum)
        nTracks=3

        trackDirR = np.zeros(trackNum)
        trackDirPhi = np.zeros(trackNum)
        trackX = np.zeros(trackNum)
        trackY = np.zeros(trackNum)

        for trk in range(nTracks) :

           #additional lines to simplify the MC for single layer try
           #jetDirR=0 #FIXME REMOVE THIS LINE FOR REAL TRAINING (it bias)

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


        #    trackDirR[trk] = random.uniform(0+jetDirR,openAngle+jetDirR)
        #    trackDirPhi[trk] = random.uniform(0,2*math.pi)
        # laycount = 0
        for lay in range(layNum) :
           for trk in range(nTracks) :
               d = (lay+1)*layDist*math.tan(trackDirR[trk])
               x = d*(math.sin(trackDirPhi[trk])+trackX[trk])
               y = d*(math.cos(trackDirPhi[trk])+trackY[trk])
            #    if x>jetDim/2 or y>jetDim/2 or y<-jetDim/2 or x<-jetDim/2 :
                #    print(x,y,trackDirR[trk], math.tan(trackDirR[trk]) )
               for electrons in range(100) :
                       channelx=int(random.gauss(x,0.3))
                       channely=int(random.gauss(y,0.3))
                       if channelx>=-jetDim/2 and channelx < jetDim/2 and channely>=-jetDim/2 and channely < jetDim/2 :
                           jetMap[lay][channelx+jetDim/2][channely+jetDim/2]+=300*1./100.
                           trackMap[trk][lay][channelx+jetDim/2][channely+jetDim/2] +=300*1./100.
               if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2  and lay==0: #TODO 10 par
            #    if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2  and lay==0: #TODO 4 par
            #             trackPar[trk][0] = x
            #             trackPar[trk][1] = y
            #    if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2  and lay==3:
            #             trackPar[trk][2] = x
            #             trackPar[trk][3] = y
            #    if x>=-jetDim/2 and x < jetDim/2 and y>=-jetDim/2 and y < jetDim/2 : #TODO 10par
            #         if lay == 0:
            #             trackPar[trk][0] = trackX[trk]
            #             trackPar[trk][1] = trackY[trk]
            #             trackPar[trk][2] = trackDirR[trk]
            #             trackPar[trk][3] = trackDirPhi[trk]
            #         trackPar[trk][2*lay+4] = x
            #         # laycount=laycount+1
            #         trackPar[trk][2*lay+1+4] = y
                    # laycount=laycount+1
                    trackPar[trk] = (x,y,trackDirR[trk],trackDirPhi[trk]) #TODO new implementation
                    # trackPar[trk] = (trackX[trk],trackY[trk],trackDirR[trk],trackDirPhi[trk]) #TODO new implementation

        for trk in range(trackNum) :
            trackObj.append((trackMap[trk],trackPar[trk], mindistMod(trackPar,trk)))

        trackObj = sorted(trackObj, key= (lambda( trackObj) : trackObj[2]), reverse=True)

        distances = np.zeros([trackNum])
        trackMapSorted= np.zeros(shape=(trackNum,layNum, jetDim,jetDim))
        # trackParSorted= np.zeros(shape=(trackNum,parNum+2*layNum)) #TODO 10par
        trackParSorted= np.zeros(shape=(trackNum,parNum)) #TODO 4 par

        for trk in range(trackNum) :
           (trackMapSorted[trk], trackParSorted[trk], distances[trk]) = trackObj[trk] #chiedi ad andrea come mai ho douto ricrearni nuovi

        #if(i<=jetNum*1) :
        if one_track_only== False :
           target_map[i] = trackMapSorted
           target_map0[i] =trackMapSorted[0] #solo prima traccia
           target_par[i] = trackParSorted
           target_par0[i] = trackParSorted[0] #solo prima traccia
           input_[i] = jetMap
        else :
           target_map[i] =trackMapSorted[0] #solo prima traccia
           target_par[i] = trackParSorted[0] #solo prima traccia
           input_[i] = jetMap

       # if(i>jetNum*0.8) :
       #     validation_target_map[i] = trackMapSorted
       #     #validation_target_par[i] = trackParSorted QUESTA e QUELLA BUONA e completa
       #     validation_target_par[i] = trackParSorted[0] #solo prima traccia
       #     validation_input_[i] = jetMap

    # outfile = file("toy_MC_event{ev}_layer{llay}_angle{angle}.npz".format(ev=jetNum,llay=layNum, angle=openAngle),"w")
    # np.savez(outfile, input_=input_, target_par=target_par, target_map=target_map, target_par0=target_par0, target_map0=target_map0)
    # # np.save(outfile, target_par)
    # np.save(outfile, target_map)

    if one_track_only :
        np.savez("ONETRACK_toy_MC_event_{ev}_layer{llay}_angle{angle}_4par_{seed}_300".format(ev=jetNum,llay=layNum, angle=openAngle, seed=seed), input_=input_, target_par=target_par, target_map=target_map)
    elif one_track :
        np.savez("ONETRACK_toy_MC_event_{ev}_layer{llay}_angle{angle}_4par_{seed}".format(ev=jetNum,llay=layNum, angle=openAngle, seed=seed), input_=input_, target_par=target_par0, target_map=target_map0)
    else :
        np.savez("toy_MC_event{ev}_layer{llay}_angle{angle}".format(ev=jetNum,llay=layNum, angle=openAngle), input_=input_, target_par=target_par, target_map=target_map)




#---------------------------------------------EXTERNAL INPUT -------------------------------//

if generate==False :
    # inputfile = file(input_name,"w")
    #(iput_, target_par, target_map) = np.load(input_name)

    # loadedfile = np.load(input_name)
    # input_= loadedfile['input_']
    # target_par0= loadedfile['target_par0']
    # target_map0= loadedfile['target_map0']
    # target_par= loadedfile['target_par']
    # target_map= loadedfile['target_map']
    if input_name=="multi" :
        files=glob.glob('/home/users/bertacch/cms/cms_mywork/trackjet/toyNN/ONETRACK_toy_MC_event_100000_layer4_angle1_4par_*.npz')
        print("Number of file: =",len(files))
        con = 0
        for f in files :
            if(con<20) :
                print("name =",f)
                loadedfile = np.load(f)
                in_= loadedfile['input_']
                tp= loadedfile['target_par']
                tm= loadedfile['target_map']
                print ("here")
                input_ = np.concatenate((input_,in_),0)
                print ("here")

                target_par = np.concatenate((target_par,tp),0)
                print ("here")

                target_map = np.concatenate((target_map,tm),0)
                con = con +1


    elif one_track :
        loadedfile = np.load("ONETRACK_"+input_name)
    else :
        loadedfile = np.load(input_name)

    if input_name != "multi" :
        input_= loadedfile['input_']
        target_par= loadedfile['target_par']
        target_map= loadedfile['target_map']

    for jet in range(jetNum) :#TODO new implementation
        for p in range(parNum) :
            target_par[jet][p]+=jetDim/2     #FIXME remove jetDim/2: only for RELU activation!!!!!
        # target_par[jet][1]+=jetDim/2     #FIXME remove jetDim/2: only for RELU activation!!!!!
        # for trk in range(trackNum) : #TODO multitrack
        #          target_par[jet][trk][0]+=jetDim/2     #FIXME remove jetDim/2: only for RELU activation!!!!!
        #          target_par[jet][trk][1]+=jetDim/2     #FIXME remove jetDim/2: only for RELU activation!!!!!


    print("loaded complete")
#-----------------------------------------KERAS MODEL -----------------------------------#

if train or predict :

        # print ("target shape", target_map.shape)
        # print ("target shape", target_par.shape)
        # print ("input shape", input_.shape)

    # model = Sequential([
    #         Conv2D(30,3, data_format="channels_first", input_shape=(layNum,jetDim,jetDim), activation='relu'),
    #         Conv2D(30,9, activation='relu'),
    #         Conv2D(3,3, activation='relu'),
    #         MaxPooling2D(pool_size=(3,3)),
    #         Flatten(),
    #         Dense(100, activation='relu'),
    #         Dense(4, activation='relu')
    #         ])
    if one_track :
        NNinputs = Input(shape=(layNum,jetDim,jetDim))
        conv30_3 = Conv2D(30,3, data_format="channels_first", input_shape=(layNum,jetDim,jetDim), activation='relu')(NNinputs)
        conv30_9 = Conv2D(30,9, data_format="channels_first", activation='relu')(conv30_3)
        conv3_3 = Conv2D(3,3, data_format="channels_first", activation='relu')(conv30_9)
        maxpool = MaxPooling2D(pool_size=(3,3))(conv3_3)
        flat = Flatten()(maxpool)
        dense100 = Dense(100, activation='relu')(flat)
        # dense4 = Dense(4, activation='relu')(dense100) #here the four track parameters
        dense4 = Dense(4, activation='relu')(dense100) #here the four track parameters #TODO 10par
        dense100 = Dense(400, activation='relu')(dense4) #784
        print (" shape", dense100.shape)
        reshaped = Reshape((layNum, 10,10))(dense100) #28(layNum, 10,10)
        deconv3_21 = Conv2DTranspose(3,21, data_format="channels_first", activation='relu', dilation_rate=2)(reshaped)
        newInput = concatenate([NNinputs,deconv3_21],axis=1)
        #conv1_1 = Conv2D(4,1,data_format="channels_first", activation='relu')(newInput)
        conv3_11 = Conv2D(3,11,data_format="channels_first", activation='relu')(newInput)
        denconv7_11 = Conv2DTranspose(7,11, data_format="channels_first", activation='relu', dilation_rate=2)(conv3_11)
        conv1_1 = Conv2D(4,1,data_format="channels_first", activation='relu')(denconv7_11)

    else :
        NNinputs = Input(shape=(layNum,jetDim,jetDim))
        conv30_3 = Conv2D(30,3, data_format="channels_first", input_shape=(layNum,jetDim,jetDim), activation='relu')(NNinputs)
        conv30_9 = Conv2D(30,9, data_format="channels_first", activation='relu')(conv30_3)
        conv3_3 = Conv2D(3,3, data_format="channels_first", activation='relu')(conv30_9)
        maxpool = MaxPooling2D(pool_size=(3,3))(conv3_3)
        flat = Flatten()(maxpool)
        dense100 = Dense(100, activation='relu')(flat)
        dense4 = Dense(4, activation='relu')(dense100) #here the four track parameters
        dense100 = Dense(100, activation='relu')(dense4) #784
        print (" shape", dense100.shape)
        reshaped = Reshape((layNum, 10,10))(dense100) #28
        deconv3_21 = Conv2DTranspose(3,21, data_format="channels_first", activation='relu', dilation_rate=2)(reshaped)
        newInput = concatenate([NNinputs,deconv3_21],axis=1)
        conv3_11 = Conv2D(3,11,data_format="channels_first", activation='relu')(newInput)
        denconv7_11 = Conv2DTranspose(7,11, data_format="channels_first", activation='relu', dilation_rate=2)(conv3_11)
        conv1_1 = Conv2D(4,1,data_format="channels_first", activation='relu')(denconv7_11)


    model = Model(NNinputs,[dense4,conv1_1]) #TODO map pred
    #model = Model(NNinputs,[conv1_1]) #TODO map pred
    # model = Model(NNinputs,[dense4])

        #model = Model(NNinputs,conv1_1)
        # model.compile('adadelta', 'mse')
    model.compile('adam', 'mse')

    model.summary()




# model = Sequential([
#         Dense(200, activation = 'relu', input_shape=(layNum,jetDim,jetDim) ),
#         Flatten(),
#         Dropout(0.2),
#         Dense(4)
#     ])

#-----------------------------------------NN TRAINING -----------------------------------#
continue_training = False
if train :
    if continue_training :
        model.load_weights('toyNN_train_{Seed}.h5'.format(Seed=seed))
        history  = model.fit(input_, [target_par, target_map],  batch_size=batch_size, nb_epoch=epochs+epochs, verbose = 2, validation_split=valSplit, initial_epoch=31) #TODO map pred
        model.save_weights('toyNN_train_bis_{Seed}.h5'.format(Seed=seed))
    else :
        history  = model.fit(input_, [target_par, target_map],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit) #TODO map pred
        #history  = model.fit(input_, [target_map],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit) #TODO map pred
        # model.fit(input_, [target_par],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit)
        model.save_weights('toyNN_train_{Seed}.h5'.format(Seed=seed))

    pylab.plot(history.history['loss'])
    pylab.plot(history.history['val_loss'])
    pylab.title('model loss')
    pylab.ylabel('loss')
    pylab.xlabel('epoch')
    pylab.legend(['train', 'test'], loc='upper right')
    pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
    pylab.show()

if predict :

    if train == False :
        model.load_weights('toyNN_train_{Seed}.h5'.format(Seed=seed))

    [validation_par,validation_map] = model.predict(input_)#TODO map pred
    #validation_map = model.predict(input_)#TODO map pred
    #validation_par = model.predict(input_)
    #outpred = file("toyNN_prediction_{Seed}".format(Seed=seed),"w")
    np.savez("toyNN_prediction_{Seed}".format(Seed=seed), validation_par=validation_par,validation_map=validation_map) #TODO map pred
    # np.savez(outpred, validation_par=validation_par)


    #model.save("toyNN_prediction.h5")
# print("------------------------")
# print("validation shape", validation.shape)
# print(validation)




#------------------------------------------------ PRINT ROOT FILE -----------------------------------#

if output :
     if predict == False :
        loadpred = np.load("toyNN_prediction_{Seed}.npz".format(Seed=seed))
        # validation_par0 = loadpred['validation_par0']
        # validation_map0 = loadpred['validation_map0']
        validation_par = loadpred['validation_par']
        validation_map = loadpred['validation_map'] #TODO map pred


     from ROOT import *
     gROOT.Reset()
     gROOT.SetBatch(True); #no draw at screen
     numPrint = 20
     validation_offset=int(jetNum*(1-valSplit)+1)

     #numPrint_last = jetNum-num PrintnumPrint_last,jetNum

     canvasTot = []
     canvasTrack = []
     mapTot = []
     mapTrack = []
     graphTargetTot = []
     graphTargetTrack = []
     graphPredTot = []
     graphPredTrack = []

     mapTargetTot = [] #TODO mapTargetTrack and pred
     mapPredTot = []

     for jet in range(numPrint) :

         mapTot_jet = []
         mapTrack_jet = []

         canvasTot_jet = []
         canvasTrack_jet = []

         graphTargetTot_jet = []
         graphTargetTrack_jet = []

         graphPredTot_jet = []
         graphPredTrack_jet = []

         mapTargetTot_jet = []
         mapPredTot_jet = []

         for lay in range(layNum) :
             mapTot_jet.append(TH2F("mapTot_%d_%d" % (jet, lay), "mapTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             canvasTot_jet.append(TCanvas("canvasTot_%d_%d" % (jet, lay), "canvasTot_%d_%d" % (jet, lay), 600,800))
             graphTargetTot_jet.append(TGraph(1))
             graphPredTot_jet.append(TGraph(1))
             mapTargetTot_jet.append(TH2F("mapTargetTot_%d_%d" % (jet, lay), "mapTargetTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             mapPredTot_jet.append(TH2F("mapPredTot_%d_%d" % (jet, lay), "mapPredTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             mapTrack_lay = []
             canvasTrack_lay = []
             graphTargetTrack_lay = []
             graphPredTrack_lay = []
             for trk in range (trackNum) :
                 mapTrack_lay.append(TH2F("mapTot_%d_%d_%d" % (jet, lay,trk), "mapTot_%d_%d_%d" % (jet, lay,trk), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
                 canvasTrack_lay.append(TCanvas("canvasTrack_%d_%d_%d" % (jet, lay, trk), "canvasTrack_%d_%d_%d" % (jet, lay, trk), 600,800))
                 graphTargetTrack_lay.append(TGraph(1))
                 graphPredTrack_lay.append(TGraph(1))
             mapTrack_jet.append(mapTrack_lay)
             canvasTrack_jet.append(canvasTrack_lay)
             graphTargetTrack_jet.append(graphTargetTrack_lay)
             graphPredTrack_jet.append(graphPredTrack_lay)
         mapTot.append(mapTot_jet)
         mapTrack.append(mapTrack_jet)
         canvasTot.append(canvasTot_jet)
         canvasTrack.append(canvasTrack_jet)
         graphTargetTot.append(graphTargetTot_jet)
         graphTargetTrack.append(graphTargetTrack_jet)
         graphPredTot.append(graphPredTot_jet)
         graphPredTrack.append(graphPredTrack_jet)

         mapTargetTot.append(mapTargetTot_jet)
         mapPredTot.append(mapPredTot_jet)


     for jet in range(numPrint) :
         j_eff = jet+validation_offset
        #  j_eff = jet
         for lay in range(layNum) :
             for x in range(jetDim) :
                 for y in range(jetDim) :
                     mapTot[jet][lay].SetBinContent(x+1,y+1,input_[j_eff][lay][x][y])
                     mapTargetTot[jet][lay].SetBinContent(x+1,y+1,target_map[j_eff][lay][x][y]) #TODO map pred
                     mapPredTot[jet][lay].SetBinContent(x+1,y+1,validation_map[j_eff][lay][x][y])  #TODO map pred
                     mapTargetTot[jet][lay].SetLineColor(2)
                     mapPredTot[jet][lay].SetLineColor(3)
                     mapTargetTot[jet][lay].SetLineWidth(3)
                     mapPredTot[jet][lay].SetLineWidth(3)
                    #  for trk in range(trackNum) :#TODO map pred
                    #       mapTrack[jet][lay][trk].SetBinContent(x+1,y+1,target_map[j_eff][trk][lay][x][y])#TODO map pred

             if 1 : #maybe if predict can be useful?
                 print("--------------------------------------------------")
                 print("jet, lay", j_eff, lay)
                #  print("target:", target_par[j_eff][0]-jetDim/2,target_par[j_eff][1]-jetDim/2, target_par[j_eff][2], target_par[j_eff][3])#TODO new implementation
                #  print("prediction:", validation_par[j_eff][0]-jetDim/2,validation_par[j_eff][1]-jetDim/2, validation_par[j_eff][2], validation_par[j_eff][3])#TODO new implementation
                 print("target:", target_par[j_eff][0]-jetDim/2,target_par[j_eff][1]-jetDim/2, target_par[j_eff][2]-jetDim/2, target_par[j_eff][3]-jetDim/2)#TODO new implementation
                 print("prediction:", validation_par[j_eff][0]-jetDim/2,validation_par[j_eff][1]-jetDim/2, validation_par[j_eff][2]-jetDim/2, validation_par[j_eff][3]-jetDim/2)#TODO new implementation

                 for i in range(1) : #trackNum

                     d = (lay+1)*layDist*math.tan(target_par[j_eff][2])#TODO new implementation
                     xoff = (target_par[j_eff][0]-jetDim/2)/(d/(lay+1))-math.sin(target_par[j_eff][3])#TODO new implementation
                     yoff = (target_par[j_eff][1]-jetDim/2)/(d/(lay+1))-math.cos(target_par[j_eff][3])#TODO new implementation
                     x = d*(math.sin(target_par[j_eff][3])+xoff) #-jetDim/2#TODO new implementation
                     y = d*(math.cos(target_par[j_eff][3])+yoff) #-jetDim/2#TODO new implementation

                    #  d = (lay+1)*layDist*math.tan(target_par[j_eff][2])#TODO new implementation
                    #  x = d*(math.sin(target_par[j_eff][3])+target_par[j_eff][0]) #-jetDim/2#TODO new implementation
                    #  y = d*(math.cos(target_par[j_eff][3])+target_par[j_eff][1]) #-jetDim/2#TODO new implementation

                    #  x = (target_par[j_eff][2]-target_par[j_eff][0])/(layNum-1)*lay+target_par[j_eff][0]-jetDim/2 #TODO 4par
                    #  y = (target_par[j_eff][3]-target_par[j_eff][1])/(layNum-1)*lay+target_par[j_eff][1]-jetDim/2 #TODO 4par

                    #  x = target_par[j_eff][2*lay+4] #TODO 10 par
                    #  y = target_par[j_eff][2*lay+1+4] #TODO 10 par
                     print("evaluated(TARGET) x,y", x,y)
                     graphTargetTot[jet][lay].SetMarkerColor(2)
                     graphTargetTot[jet][lay].SetMarkerStyle(2)
                     graphTargetTot[jet][lay].SetMarkerSize(3)
                     graphTargetTot[jet][lay].SetPoint(i,x,y)

                 for i in range(1) : #trackNum
                     d = (lay+1)*layDist*math.tan(validation_par[j_eff][2])
                    #  print("R=",validation_par[j_eff][2])
                    #  print("tanR=",math.tan(validation_par[j_eff][2]))
                    #  print("(lay+1)*layDist=",(lay+1)*layDist)
                    #   print("d=",d)
                     if(d!=0.0) :#TODO new implementation
                         xoff = (validation_par[j_eff][0]-jetDim/2)/(d/(lay+1))-math.sin(validation_par[j_eff][3])
                         yoff = (validation_par[j_eff][1]-jetDim/2)/(d/(lay+1))-math.cos(validation_par[j_eff][3])
                         x = d*(math.sin(validation_par[j_eff][3])+xoff) #-jetDim/2
                         y = d*(math.cos(validation_par[j_eff][3])+yoff) #-jetDim/2
                     else :#TODO new implementation
                         x=0
                         y=0
                    #  d = (lay+1)*layDist*math.tan(validation_par[j_eff][2])#TODO new implementation
                    #  x = d*(math.sin(validation_par[j_eff][3])+validation_par[j_eff][0]) #-jetDim/2#TODO new implementation
                    #  y = d*(math.cos(validation_par[j_eff][3])+validation_par[j_eff][1]) #-jetDim/2#TODO new implementation
                    #  x = d*math.sin(validation_par[j_eff][3]+validation_par[j_eff][0]-jetDim/2)
                    #  y = d*math.cos(validation_par[j_eff][3]+validation_par[j_eff][1]-jetDim/2)


                    #  x = (validation_par[j_eff][2]-validation_par[j_eff][0])/(layNum-1)*lay+validation_par[j_eff][0]-jetDim/2  #TODO 4par
                    #  y = (validation_par[j_eff][3]-validation_par[j_eff][1])/(layNum-1)*lay+validation_par[j_eff][1]-jetDim/2  #TODO 4par


                    #  x = validation_par[j_eff][2*lay+4] #TODO 10 par
                    #  y = validation_par[j_eff][2*lay+1+4] #TODO 10 par
                     print("evaluated(pred) x,y", x,y)
                     graphPredTot[jet][lay].SetMarkerColor(3)
                     graphPredTot[jet][lay].SetMarkerStyle(28)
                     graphPredTot[jet][lay].SetMarkerSize(3)
                     graphPredTot[jet][lay].SetPoint(i,x,y)


             #mapTot[jet][lay].SetBinContent((int)(validation[jet][0]+jetDim/2),(int)(validation[jet][1]+jetDim/2),1)
                        # if validation[jet][trk][lay][x][y]>0.5 :
                        #     mapTot[jet][lay].SetBinContent(x+1,y+1,10*validation[jet][trk][lay][x][y])

     output_file = TFile("toyNN_{Seed}.root".format(Seed=seed),"recreate")
    #  for jet in range(10) :
    #      for lay in range(layNum) :
    #          mapTot[jet][lay].Write()
    #          for trk in range(trackNum) :
    #              mapTrack[jet][lay][trk].Write()

     for jet in range(numPrint) :
         for lay in range(layNum) :
             canvasTot[jet][lay].cd()
             mapTot[jet][lay].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].GetYaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].Draw("colz") #colz
             mapTargetTot[jet][lay].Draw("same box")

             mapPredTot[jet][lay].Draw("same box")

             graphTargetTot[jet][lay].Draw("SAME P")
             graphPredTot[jet][lay].Draw("SAME P")
            #  graphPredTot[jet][lay].Draw()
             canvasTot[jet][lay].Write()
            #  for trk in range(trackNum) :
            #       mapTrack[jet][lay][trk].Write()
     output_file.Close()


# if False :
#     # Path to the data txt file on disk.
#     data_path = 'data.txt'
#
#     # Vectorize the data.
#     input_characters = set()
#     #target_characters = set()
#
#     #input_characters = sorted(list(input_characters))
#     #target_characters = sorted(list([x for x in range(32)]))
#     #num_encoder_tokens = len(input_characters)
#     num_decoder_tokens =1 # len(target_characters)
#     #max_encoder_seq_length = max([len(txt) for txt in input_])
#     max_decoder_seq_length = slen+1
#
#     print('Number of samples:', len(input_))
#     print('Number of unique output tokens:', num_decoder_tokens)
#     print('Max sequence length for outputs:', max_decoder_seq_length)
#
#     #input_token_index = dict(
#     #    [(char, i) for i, char in enumerate(input_characters)])
#     #target_token_index = dict(
#     #    [(char, i) for i, char in enumerate(target_characters)])
#
#     fixed_input_data = np.zeros(
#        (len(input_), layNum, JetDim,JetDim),
#        dtype='float32')
#     decoder_input_data = np.zeros(
#        (len(input_), max_decoder_seq_length, num_decoder_tokens*2),
#        dtype='float32')
#     decoder_target_data = np.zeros(
#        (len(input_), max_decoder_seq_length, num_decoder_tokens*2),
#        dtype='float32')
#
#     for i, (input_, target_map) in enumerate(zip(input_, target_maps)):
#     #   for t, char in enumerate(input_):
#     #       encoder_input_data[i, t, input_token_index[char]] = 1.
#        for j in range(JetDim):
#          for k in range(JetDim):
#            fixed_input_data[i,j,k]=input_[j][k]
#        for t, char in enumerate(target_map):
#            # decoder_target_data is ahead of decoder_input_data by one timestep
#            decoder_input_data[i, t,0] =  char[0]
#            decoder_input_data[i, t,1] =  char[1]
#            if t > 0:
#                # decoder_target_data will be ahead by one timestep
#                # and will not include the start character.
#                decoder_target_data[i, t - 1,0] =  char[0]
#                decoder_target_data[i, t - 1,1] =  char[1]
#
#     print(decoder_input_data)
#     print(fixed_input_data)
#
#     # Define an input sequence and process it.
#     fixed_inputs = Input(shape=( jetDim,jetDim,))
#     flat = Flatten()(fixed_inputs)
#     tofeed_as_state=  Dense(latent_dim, kernel_initializer="lecun_uniform", activation="relu")(flat)#(fixed_inputs)
#     tofeed_as_state2=  Dense(latent_dim, kernel_initializer="lecun_uniform", activation="relu")(tofeed_as_state)
#
#     #encoder = LSTM(latent_dim, return_state=True)
#     #encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#     # We discard `encoder_outputs` and only keep the states.
#     encoder_states = [tofeed_as_state,tofeed_as_state2]
#
#     # Set up the decoder, using `encoder_states` as initial state.
#     decoder_inputs = Input(shape=(None,2))
#     # We set up our decoder to return full output sequences,
#     # and to return internal states as well. We don't use the
#     # return states in the training model, but we will use them in inference.
#     decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
#     #print(decoder_inputs,fixed_input_data)
#     #inp=np.concatenate((decoder_inputs,fixed_input_data),axis=1)
#     #print(inp)
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                         initial_state=encoder_states)
#     decoder_dense = Dense(num_decoder_tokens*2, activation='relu') #was softmax
#     decoder_outputs = decoder_dense(decoder_outputs)
#
#     # Define the model that will turn
#     # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#     model = Model([fixed_inputs, decoder_inputs], decoder_outputs)
#
#     # Run training
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.summary()
#     #model= load_model('reg.h5')
#     #odel.summary()
#
#     model.load_weights('trained130.h5')
#
#     model.fit([fixed_input_data, decoder_input_data], decoder_target_data,
#             callbacks=[wH],
#             batch_size=batch_size,
#              epochs=epochs,
#              validation_split=0.2)
#     model.save('sillysort2D.h5')
#     # Save model
#
#     # Next: inference mode (sampling).
#     # Here's the drill:
#     # 1) encode input and retrieve initial decoder state
#     # 2) run one step of decoder with this initial state
#     # and a "start of sequence" token as target.
#     # Output will be the next target token
#     # 3) Repeat with the current target token and current states
#
#     # Define sampling models
#     encoder_model = Model(fixed_inputs, encoder_states)
#
#     decoder_state_input_h = Input(shape=(latent_dim,))
#     decoder_state_input_c = Input(shape=(latent_dim,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, state_h, state_c = decoder_lstm(
#        decoder_inputs, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = Model(
#        [decoder_inputs] + decoder_states_inputs,
#        [decoder_outputs] + decoder_states)
#
#     decoder_model.summary()
#     encoder_model.summary()
#
#
#     # Reverse-lookup token index to decode sequences back to
#     # something readable.
#     #reverse_input_char_index = dict(
#     #    (i, char) for char, i in input_token_index.items())
#     #reverse_target_char_index = dict(
#     #    (i, char) for char, i in target_token_index.items())
#
#
#     def decode_sequence(input_seq):
#        # Encode the input as state vectors.
#        states_value = encoder_model.predict(input_seq)
#
#        # Generate empty target sequence of length 1.
#        target_seq = np.zeros((1, 1, num_decoder_tokens*2))
#        # Populate the first character of target sequence with the start character.
#        target_seq[0, 0,0] = -10
#        target_seq[0, 0,1] = -10
#
#
#        # Sampling loop for a batch of sequences
#        # (to simplify, here we assume a batch of size 1).
#        stop_condition = False
#        decoded_sentence = []
#        while not stop_condition:
#            output_tokens, h, c = decoder_model.predict(
#                [target_seq] + states_value)
#
#            # Sample a token
#            sampled_char_ndarray = output_tokens[0, -1, :]
#            sampled_char=sampled_char_ndarray
#            #print( sampled_char, sampled_char[0])
#            decoded_sentence.append( sampled_char)
#
#            # Exit condition: either hit max length
#            # or find stop character.
#            if (sampled_char[0] >= 35 or sampled_char[1] >= 35 or
#               len(decoded_sentence) > max_decoder_seq_length):
#                stop_condition = True
#
#            # Update the target sequence (of length 1).
#            target_seq[0,0,0] = sampled_char[0]
#            target_seq[0,0,1] = sampled_char[1]
#     #, num_decoder_tokens))
#     #       target_seq[0, 0, sampled_char] = 1.
#
#            # Update states
#            states_value = [h, c]
#
#        return decoded_sentence
#
#
#     for seq_index in range(10):
#        # Take one sequence (part of the training test)
#        # for trying out decoding.
#        input_seq = fixed_input_data[seq_index: seq_index + 1]
#        decoded_sentence = decode_sequence(input_seq)
#        target_map=target_maps[seq_index]
#        print('-')
#        print('Input sentence:', input_[seq_index])
#        print('Target sentence:',  target_map)
#        print('Decoded sentence:', decoded_sentence)
