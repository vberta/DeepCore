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
from keras.initializers import *
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

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from numpy import concatenate as concatenatenp

#######################################
#
# USAGE: python yoloJet.py --seed SEED --convert --input INPUT --training --predict --output
#
######################################


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


########################internal parameters########################
standard_mse = False
noProb = False
uproot_flag = True
numPrint = 20
continue_training = True
no_pt_eta = False
ROI_loss = True
outEvent= 2 #complete plots for this only


batch_size = 64#64#128#32 # Batch size for training. //32 is the good
epochs = 35 # Number of epochs to train for.
start_epoch = 350 #260 sono senza dropout
latent_dim = 70 # Latent dimensionality of the encoding space.

valSplit=0.2
prob_thr =0.9

jetNum=16460#1000#16460#200000#100000
jetNum_validation = 3441
jetDim=30
trackNum =3# 10
genTrackNum=3
layNum = 4
parNum = 4




import random
random.seed(seed)



# if(predict or output) :
#     input_ = np.zeros(shape=(jetNum, jetDim,jetDim,layNum)) #jetMap
#     target_ = np.zeros(shape=(jetNum,jetDim, jetDim,trackNum,parNum+1))#+1
#     target_prob = np.zeros(shape=(jetNum,jetDim,jetDim,trackNum))

jetNum_test=50
input_test = np.zeros(shape=(jetNum_test, jetDim,jetDim,layNum)) #jetMap
target_test = np.zeros(shape=(jetNum_test,jetDim, jetDim,trackNum,parNum+1))#+1
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
            pylab.ylabel('entries')
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




_EPSILON = 1e-7

def _to_tensor(x, dtype):
    return ops.convert_to_tensor(x, dtype=dtype)

def epsilon():
    return _EPSILON

def loss_weighted_crossentropy(target, output):
    epsilon_ = _to_tensor(keras.backend.epsilon(), output.dtype.base_dtype)
    # epsilon_ = keras.backend.epsilon()
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output / (1 - output))
    retval = nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=125)#900=works #2900=200x200, 125=30x30
    return K.mean(retval, axis=-1)

#
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
    # return K.mean(retval, axis=-1)

# def loss_weighted_crossentropy(target, output):
#     # epsilon_ = keras.backend.epsilon()
#     # output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
#     # output = math_ops.log(output / (1 - output))
#     wBCE = nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=2900)#2900 #900=works
#     return K.mean(wBCE, axis=-1)


# def loss_mse_weight(y_weight) : #NOT USED!!!!
#     _epsilon = K.epsilon()
#     def loss_mse(y_true, y_pred):
#         # y_pred = np.clip(y_pred, _epsilon, 1.0-_epsilon)
#         y_pred = clip_ops.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
#         out = K.square(y_pred - y_true)*(y_weight)
#         return K.mean(out, axis=-1)
#     return loss_mse
#altra idea: aggiungi vettore al target e poi lo splitti dentro la loss in 2 sotto vettori e fai l'mse di quelli.

def loss_mse_select(y_true, y_pred) :
    # _epsilon = K.epsilon()
    # y_pred = clip_ops.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    wei = y_true[:,:,:,:,-1:]
    pred = y_pred[:,:,:,:,:-1]
    true =  y_true[:,:,:,:,:-1]

    # pred = tf.transpose(pred, perm = [4,0,1,2,3])
    # true = tf.transpose(true, perm = [4,0,1,2,3])
    # wei = K.expand_dims(wei,0)
    out =K.square(pred-true)*wei
    # out = tf.transpose(out, perm =[1,2,3,4,0])
    # print("len wei=",(tf.reduce_sum(wei,axis=None)*4))
    return tf.reduce_sum(out, axis=None)/(tf.reduce_sum(wei,axis=None)*4) #4=numPar
    # return K.mean(out, axis=-1)

def loss_mse_select_clipped(y_true, y_pred) :
    wei = y_true[:,:,:,:,-1:]
    pred = y_pred[:,:,:,:,:-1]
    true =  y_true[:,:,:,:,:-1]
    out =K.square(tf.clip_by_value(pred-true,-5,5))*wei
    return tf.reduce_sum(out, axis=None)/(tf.reduce_sum(wei,axis=None)*4) #4=numPar


def Generator(files) :
    while 1:
        for f in files :
            import uproot
            tfile = uproot.open(f)
            # print("file=",f)

            tree = tfile["demo"]["NNClustSeedInputTree"]
            input_ = tree.array("cluster_measured")
            input_jeta = tree.array("jet_eta")
            input_jpt = tree.array("jet_pt")
            target_ = tree.array("trackPar")
            target_prob = tree.array("trackProb")

            if(standard_mse) :
                target_= target_[:,:,:,:,:-1]

            if(ROI_loss) :
                wei = target_[:,:,:,:,-1:]
                nev = len(target_prob)
                target_prob = np.reshape(target_prob, (nev,jetDim,jetDim,trackNum,1))
                target_prob = concatenatenp([target_prob,wei],axis=4)
            #without flag for standard mse
            # if(standard_mse) :
            #     target_ = np.zeros(shape=(len(target_5par),jetDim, jetDim,trackNum,parNum))
            #     for j in range(len(target_5par)) :
            #         for x in range(jetDim) :
            #             for y in range(jetDim) :
            #                 for trk in range (trackNum) :
            #                     for p in range (parNum) :
            #                         target_[j][x][y][trk][p] = target_5par[j][x][y][trk][p]
            # else :
            #     target_ = target_5par

            # print("file dimension=", len(input_jeta))

            for k in range(len(input_jeta)/batch_size) :
                # print("range=", k)
                if(not noProb) :
                    yield [input_[batch_size*(k):batch_size*(k+1)],input_jeta[batch_size*(k):batch_size*(k+1)],input_jpt[batch_size*(k):batch_size*(k+1)]], [target_[batch_size*(k):batch_size*(k+1)],target_prob[batch_size*(k):batch_size*(k+1)]]
                else :
                    yield [input_[batch_size*(k):batch_size*(k+1)],input_jeta[batch_size*(k):batch_size*(k+1)],input_jpt[batch_size*(k):batch_size*(k+1)]], target_[batch_size*(k):batch_size*(k+1)]


def prop_on_layer(x1,y1,eta,phi,eta_jet,lay) :

    theta_jet = 2*math.atan(math.exp(-eta_jet))
    eta = eta+eta_jet
    theta = 2*math.atan(math.exp(-eta))

    # print("x1,y1=",x1,y1)
    # print("delta eta, eta", eta, eta_jet)
    # print("delta phi", phi)
    # print("theta, theta_jet", theta,theta_jet)

    if(lay==0) :
        dist=3-6.8
    if(lay==1) :
        dist=6.8-6.8
    if(lay==2) :
        dist=10.2-6.8
    if(lay==3) :
        dist=16-6.8
    distx=dist/0.01
    disty=dist/0.015

    # y_out = disty*math.sin(theta)/(math.sin(theta_jet)*math.sin(theta+theta_jet))+y1
    y_out = disty*math.sin(theta-theta_jet)/(math.sin(theta_jet)*math.sin(theta))+y1
    x_out = distx*math.tan(-phi)+x1

    # print("xout,yout",x_out/disty, y_out/disty)

    return (x_out,y_out)
#--------------------------------------------- INPUT from ROOT conversion-------------------------------#
gpu=True
chain = True
if convert :

    import ROOT
    from ROOT import TChain, TSelector, TTree, TString
    from root_numpy import *
    from root_numpy import testdata

    import gzip
    import cPickle

    tfile = ROOT.TFile(input_name)
    tree = tfile.Get('demo/NNPixSeedInputTree')

    # if(chain) :
    #     # inputChain = TChain("iC")
    #     tree = TChain('demo/NNPixSeedInputTree')
    #     # xrd =TString("root://cms-xrd-global.cern.ch//store/user/vbertacc/")
    #     print("chain start")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_1.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_2.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_3.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_4.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_5.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_6.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_7.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_8.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_9.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_10.root")
    #     tree.Add("root://cms-xrd-global.cern.ch//store/user/vbertacc/NNPixSeedInput/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NNPixSeedInput/180716_161507/0000/histo_10k_11.root")
    #
    #     # tree = inputChain.Get('demo/NNPixSeedInputTree')
    #
    #     print("chain loaded")




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

    npTree={}
    npTree["input_"]=input_
    npTree["input_jeta"]=input_jeta
    npTree["input_jpt"]=input_jpt
    npTree["target_"]=target_
    npTree["target_prob"]=target_prob


    # print("start creation weight")
    # target_loss_w = np.zeros(shape=(int(len(input_)),jetDim, jetDim,trackNum,parNum))#+1
    # for j in range (int(len(input_))) :
    #     for x in range(jetDim) :
    #         for y in range(jetDim) :
    #             for par in range(parNum) :
    #                 for trk in range(trackNum) :
    #                     if target_[j][x][y][trk][0] == 0.0  and target_[j][x][y][trk][1] == 0.0  and target_[j][x][y][trk][2] == 0.0  and target_[j][x][y][trk][3] == 0.0 :
    #                         target_loss_w[j][x][y][trk][par] = 0.0
    #                     else :
    #                         target_loss_w[j][x][y][trk][par] = 1.0
    #
    # print("end creation weight")



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

        fp=gzip.open("NNPixSeed_event_{ev}.dmpz".format(ev=jetNum),"wb")
        # cPickle.dump(input_,fp)
        # cPickle.dump(input_jeta,fp)
        # cPickle.dump(input_jpt,fp)
        # cPickle.dump(target_,fp)
        # cPickle.dump(target_prob,fp)
        cPickle.dump(npTree,fp)

        fp.close()
        # np.savez("NNPixSeed_event_{ev}".format(ev=jetNum), input_=input_, input_jeta=input_jeta, input_jpt=input_jpt, target_=target_, target_prob =target_prob, target_loss_w=target_loss_w)
        # np.savez("NNPixSeed_event_{ev}".format(ev=jetNum), input_=input_, input_jeta=input_jeta, input_jpt=input_jpt, target_=target_, target_prob =target_prob)

    print("saving data: completed")









#---------------------------------------------numpy INPUT -------------------------------#
if convert==False and gpu==True:

    old_loading= False
    if(old_loading and (predict or output)) :
        print("loading data: start")

        # import gzip
        # import pickle
        loadedfile = np.load(input_name)

        # fp=gzip.open("NNPixSeed_event_{ev}.dmpz".format(ev=jetNum),"r")
        # loadedfile = pickle.load(fp)

        input_= loadedfile['input_']
        input_jeta= loadedfile['input_jeta']
        input_jpt= loadedfile['input_jpt']
        target_= loadedfile['target_']
        target_prob= loadedfile['target_prob']
        # target_loss_w = loadedfile['target_loss_w']

        # for jet in range(jetNum) :
        #     for trk in range(trackNum) :
        #         target_[jet][trk][0]+=jetDim/2
        #         target_[jet][trk][1]+=jetDim/2





        print("loading data: completed")


    print("uproot flag=",uproot_flag)
    if(uproot_flag) :

        print("loading data: start")

        import uproot
        # tfile = uproot.open("/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/histo_10k_1.root")
        tfile = uproot.open(input_name)
        tree = tfile["demo"]["NNClustSeedInputTree"]
        input_ = tree.array("cluster_measured")
        input_jeta = tree.array("jet_eta")
        input_jpt = tree.array("jet_pt")
        target_ = tree.array("trackPar")
        target_prob = tree.array("trackProb")

        if(standard_mse) :
            print("slicing...")
            target_ = target_[:,:,:,:,:-1]

        if(ROI_loss) :
            wei = target_[:,:,:,:,-1:]
            nev = len(target_prob)
            target_prob = np.reshape(target_prob, (nev,jetDim,jetDim,trackNum,1))
            target_prob = concatenatenp([target_prob,wei],axis=4)

        # if(standard_mse) : //too slow! not used
        #     target_ = np.zeros(shape=(len(target_5par),jetDim, jetDim,trackNum,parNum))
        #     for j in range(len(target_5par)) :
        #         for x in range(jetDim) :
        #             for y in range(jetDim) :
        #                 for trk in range (trackNum) :
        #                     for p in range (parNum) :
        #                         target_[j][x][y][trk][p] = target_5par[j][x][y][trk][p]
        # else :
        #     target_ = target_5par

        # for f in range (1,11) :
        #     print("loop=", f)
        #
        #     print("input shape=", input_.shape)
        #     print("input len=", len(input_))
        #     print("input eta shape=", input_jeta.shape)
        #     print("input eta len=", len(input_jeta))
        #     print("input pt shape=", input_jpt.shape)
        #     print("input pt len=", len(input_jpt))
        #     print("target par shape=", target_.shape)
        #     print("target par len=", len(target_))
        #     print("target prob shape=", target_prob.shape)
        #     print("target prob len=", len(target_prob))
        #
        #
        #     tfile_f = uproot.open("/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/histo_10k_{n}.root".format(n=f+1))
        #     tree_f = tfile["demo"]["NNClustSeedInputTree"]
        #     input__f = tree_f.array("cluster_measured")
        #     input_jeta_f = tree_f.array("jet_eta")
        #     input_jpt_f = tree_f.array("jet_pt")
        #     target__f = tree_f.array("trackPar")
        #     target_prob_f = tree_f.array("trackProb")
        #
        #     input_ = np.concatenate((input_,input__f),axis=0)
        #     input_jeta = np.concatenate((input_jeta,input_jeta_f),axis=0)
        #     input_jpt = np.concatenate((input_jpt,input_jpt_f),axis=0)
        #     target_ = np.concatenate((target_,target__f),axis=0)
        #     target_prob = np.concatenate((target_prob,target_prob_f),axis=0)

        print("loading data: completed")




    test_sample_creation = False
    if test_sample_creation == True:
        print("testing sample creation: ...")
        for jj in range (jetNum_test) :
            j = jj+(int(len(input_))-jetNum_test-5)
            input_jeta_test[jj] = input_jeta[j]
            input_jpt_test[jj] = input_jpt[j]
            for x in range(jetDim) :
                for y in range(jetDim) :
                    for par in range(parNum+1) :
                        if(par<4) :
                            input_test[jj][x][y][par] = input_[j][x][y][par]
                        for trk in range(trackNum) :
                            target_test[jj][x][y][trk][par] = target_[j][x][y][trk][par]
                            target_prob_test[jj][x][y][trk] = target_prob[j][x][y][trk]
        print("... save ...")
        np.savez("NNPixSeed_event_{ev}_test".format(ev=jetNum_test), input_=input_test, input_jeta=input_jeta_test, input_jpt=input_jpt_test, target_=target_test, target_prob =target_prob_test)
        print("..completed")


    average_1_eval = False
    if(average_1_eval==True) :
        aver1=0
        print("evaluation of the number of 1")
        for j in range (int(len(input_))) :
            averjet = 0
            aver0jet = 0
            for x in range(jetDim) :
                for y in range(jetDim) :
                        if(ROI_loss) :
                            if(target_prob[j][x][y][0][1]) :
                                aver0jet = aver0jet+1
                                if(target_prob[j][x][y][0][0]==1) :
                                    averjet = averjet+1
                        # for trk in range(trackNum) :
                        else:
                            aver0jet=aver0jet+1
                            if target_prob[j][x][y][0]==1 :
                                averjet = averjet+1
            # print("number of 1 in jet ", j, " =", averjet)
            averjet = float(averjet)/float(aver0jet)
            aver1= aver1+averjet
        aver1 = float(aver1)/float(len(input_))
        print("average of the number of 1", aver1)
        print("Multiplicative factor to 1", 1/aver1)


    Deb1ev = False #1 event debugging (1ev input here)
    if(Deb1ev) :
            print("pre=",len(input_))
            input_= input_[1:2]
            input_jeta= input_jeta[1:2]
            input_jpt= input_jpt[1:2]
            target_= target_[1:2]
            target_prob= target_prob[1:2]
            print("post=",len(input_))

    averageADC = False #average value of ADC count input
    if(averageADC==True) :
            averADC=0
            norm = 0
            bins = []
            print("evaluation of averge ADC count")
            for j in range (int(len(input_))) :
                # averjet = 0
                for x in range(jetDim) :
                    for y in range(jetDim) :
                        for l in range(layNum) :
                            if(input_[j][x][y][l]!=0) :
                                averADC=averADC+input_[j][x][y][l]
                                bins.append(input_[j][x][y][l])
                                norm = norm +1
            averADC = float(averADC)/float(norm)
            occupancy = float(norm)/(float(jetDim*jetDim*layNum*int(len(input_))))
            print("average value of input=", averADC)
            print("average occupancy=", occupancy)

            plt.figure()
            pylab.hist(bins,100, facecolor='green')
            pylab.title('Non-zero ADC count distribution')
            pylab.ylabel('entries')
            pylab.xlabel('ADC count')
            plt.grid(True)
            pylab.savefig("ADC_count.pdf")


    # for jj in range (1) :
    #     for x in range(jetDim) :
    #         for y in range(jetDim) :
    #             for par in range(parNum+1) :
    #                 for trk in range(trackNum) :
    #                     if(target_[jj][x][y][trk][par]!=0) :
    #                         print ("x,y,trk,par",x,y,trk,par, target_[jj][x][y][trk][par])


    # testDim = target_[0,77:79,67:69,0,0:2]
    # print("testDim=", testDim)
    # print("targe 0,77,67,0,0",  target_[0][77][67][0][0])
    # fuffa = target_[0,77:79,67:69,0,0:2]
    # fuffa[0][0][0] = 0.1
    # fuffa[0][0][1] = 0.4
    # # fuffa[0][0][2] = -0.3
    # # fuffa[0][0][3] = -0.5
    # fuffa[0][1][1] = 2
    # # fuffa[0][1][2] = 4
    # print("fuffa=", fuffa)
    # print("targe 0,77,67,0,0",  target_[0][77][67][0][0])
    # t1 = np.zeros(shape=(2, 2,3))
    # t2 =  np.zeros(shape=(2, 2))
    # t1[0][0][0] = 3
    # t1[0][1][0] = 2
    # t1[1][0][0] = 1
    # t1[1][1][0] = 2.5
    # t2[0][0] = 5
    # t2[0][1] = 7
    # print("t1=", t1)
    # print("t2=",t2)
    # # t1 = K.reshape(t1,(3,2,2))
    # t1 = tf.transpose(t1, perm=[2,0,1])
    #
    # t2=K.expand_dims(t2,0)
    #
    # reess = K.square(t1-t2)
    # # reess = K.square(t2-t1)
    # # print("222",testDim[2,2,2])
    # # testDim = K.expand_dims(testDim,3)
    # # print("expandend", testDim)
    # sess = tf.InteractiveSession()
    # print(sess.run(reess))

    # print(sess.run(testDim))
    # print(sess.run(reess))
    # a = tf.Print(a, [a], message="This is testDim: ")
    # print("222",testDim[2,2,2,0])

#-----------------------------------------KERAS MODEL -----------------------------------#

if train or predict :

    from keras.layers import AlphaDropout
    #
    # if(no_pt_eta) :
    #     ComplInput = Input(shape=(jetDim,jetDim,layNum))

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
    if(not no_pt_eta) :
        ComplInput = concatenate([NNinputs,jetUps],axis=3)
    else :
        ComplInput = NNinputs
    print("ComplInput=", ComplInput.shape)
        # print("ComplInput size", ComplInput.size)















    #YoloJet
    # conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    # conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    # conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2) #(12,3)
    # conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3)#(12,3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # if(not noProb) :
    #     conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    #     reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)


    #WORKING
    # conv30_9 = Conv2D(500,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(100,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # drop2l = Dropout(0.2)(conv30_7) #only with 100k
    # conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(drop2l)#(conv30_7)#
    # conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # drop7l = Dropout(0.3)(conv15_3_1)
    # conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(drop7l)
    # drop8l = Dropout(0.3)(conv15_3_2) #new
    # if(standard_mse):
    #     conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    # else :
    #     conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # if(not noProb) :
    #     conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='relu', padding="same")(conv15_5)  #new
    #     # drop7lb = Dropout(0.6)(conv12_3_1)
    #     conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='relu', padding="same")(conv12_3_1) #drop7lb   #new
    #     conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='relu',padding="same")(conv1_3_2) #new
    #     conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    #     reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)
    #
    #     conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    #     reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)
    #
    #
    #

    # non deep NN
    # conv30_9 = Conv2D(500,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv30_9) #(12,3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv30_9)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)

    # WORKING NO Dropout
    conv30_9 = Conv2D(50,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)#20 instead of 50
    conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # drop1 = Dropout(0.1)(conv30_7)
    conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)#(conv30_7)#
    conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)

    conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # drop2 = Dropout(0.2)(conv15_3_1)
    conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)#(conv15_3_1)
    if(standard_mse):
        conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2) #(12,3)
        conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
        reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    else :
        conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2) #(12,3)
        conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
        reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)

    if(not noProb) :
        conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='relu', padding="same")(conv15_5)  #new
        conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='relu', padding="same")(conv12_3_1) #drop7lb   #new
        conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='relu',padding="same")(conv1_3_2) #new
        if(ROI_loss) :
            conv1_3_1 = Conv2D(6,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
            reshaped_prob = Reshape((jetDim,jetDim,trackNum,2))(conv1_3_1)
        else :
            conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
            reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)



    #
    # SELU
    # conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='selu', kernel_initializer= 'lecun_normal', padding="same")(ComplInput)
    # conv30_7 = Conv2D(20,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv30_9)
    # drop2l = AlphaDropout(0.2)(conv30_7)
    # conv30_5 = Conv2D(20,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(drop2l)#(conv30_7)#(drop2l)
    # drop3l = AlphaDropout(0.3)(conv30_5) #removed bydefault
    # conv20_5 = Conv2D(18,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(drop3l)#(conv30_5)
    # conv15_5 = Conv2D(15,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)
    # drop7l = AlphaDropout(0.4)(conv15_3_1) #0.4 with anubi_4l
    # conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop7l)
    # drop8l = AlphaDropout(0.4)(conv15_3_2) #new (0.3 bydefault)
    #
    # if(standard_mse):
    #     conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    # else :
    #     conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)  #new
    # conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv12_3_1) #drop7lb   #new
    # conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv1_3_2) #new
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)

# selu no dropout //QUESTA e QUELLA USATA PRE CLUSTER CENTER --->QUESTA VA MA OVERFITTA
    # conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='selu', kernel_initializer= 'lecun_normal', padding="same")(ComplInput)
    # conv30_7 = Conv2D(20,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv30_9)
    # conv30_5 = Conv2D(20,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv30_7)#(conv30_7)#(drop2l)
    # conv20_5 = Conv2D(18,5, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv30_5)#(conv30_5)
    # conv15_5 = Conv2D(15,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)
    # conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_3_1)
    #
    # if(standard_mse):
    #     conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_3_2) #(12,3)
    #     conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    # else :
    #     conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_3_2) #(12,3)
    #     conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)  #new
    # conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv12_3_1) #drop7lb   #new
    # conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv1_3_2) #new
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)

    #SELU COMPLEX
    # conv30_9 = Conv2D(100,31, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='selu', kernel_initializer= 'lecun_normal', padding="same")(ComplInput)
    # conv30_7 = Conv2D(50,21, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv30_9)
    # drop2l = AlphaDropout(0.2)(conv30_7)
    # conv30_5 = Conv2D(30,15, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(drop2l)#(conv30_7)#(drop2l)
    # drop3l = AlphaDropout(0.2)(conv30_5) #removed bydefault
    # conv20_5 = Conv2D(20,13, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(drop3l)#(conv30_5)
    # conv15_5 = Conv2D(18,9, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,7, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)
    # drop7l = AlphaDropout(0.2)(conv15_3_1) #0.4 with anubi_4l
    # conv15_3_2 = Conv2D(15,5, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop7l)
    # drop8l = AlphaDropout(0.2)(conv15_3_2) #new (0.3 bydefault)
    #
    # if(standard_mse):
    #     conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    # else :
    #     conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='selu', kernel_initializer= 'lecun_normal', padding="same")(drop8l) #(12,3)
    #     conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3) #(12,3)
    #     reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv15_5)  #new
    # conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal', padding="same")(conv12_3_1) #drop7lb   #new
    # conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='selu', kernel_initializer= 'lecun_normal',padding="same")(conv1_3_2) #new
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)






    #ORIGINAL
    # conv30_9 = Conv2D(20,7, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # conv30_5 = Conv2D(20,5, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    # conv20_5 = Conv2D(18,5, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(15,3, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # conv15_3_2 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    # conv15_3_3 = Conv2D(12,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_2)
    # conv15_3 = Conv2D(12,3, data_format="channels_last",padding="same")(conv15_3_3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum))(conv15_3)
    #
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)




    #### VERY HIGH DIMENSONS
    # conv30_9 = Conv2D(100,31, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(50,21, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # drop2l = Dropout(0.2)(conv30_7) #only with 100k
    # conv30_5 = Conv2D(30,15, data_format="channels_last", activation='relu',padding="same")(drop2l)
    # conv20_5 = Conv2D(20,13, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(18,9, data_format="channels_last", activation='relu',padding="same")(conv20_5)
    #
    # conv15_3_1 = Conv2D(15,7, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # drop7l = Dropout(0.3)(conv15_3_1)
    # conv15_3_2 = Conv2D(15,5, data_format="channels_last",activation='relu', padding="same")(drop7l)
    # drop8l = Dropout(0.3)(conv15_3_2) #new
    # conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(drop8l)
    # conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # conv12_3_1 = Conv2D(12,3, data_format="channels_last", activation='relu', padding="same")(conv15_5)  #new
    # conv1_3_2 = Conv2D(9,3, data_format="channels_last", activation='relu', padding="same")(conv12_3_1) #drop7lb   #new
    # conv1_3_3 = Conv2D(7,3, data_format="channels_last", activation='relu',padding="same")(conv1_3_2) #new
    # conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv1_3_3)
    # reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)

#complex few layer nodrop
    # conv30_9 = Conv2D(100,31, data_format="channels_last", input_shape=(jetDim,jetDim,layNum+2), activation='relu',padding="same")(ComplInput)
    # conv30_7 = Conv2D(50,21, data_format="channels_last", activation='relu',padding="same")(conv30_9)
    # # conv30_5 = Conv2D(30,15, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    # # conv20_5 = Conv2D(20,13, data_format="channels_last", activation='relu',padding="same")(conv30_5)
    # conv15_5 = Conv2D(18,9, data_format="channels_last", activation='relu',padding="same")(conv30_7)
    #
    # conv15_3_1 = Conv2D(15,7, data_format="channels_last",activation='relu', padding="same")(conv15_5)
    # # conv15_3_2 = Conv2D(15,5, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    # conv15_3_3 = Conv2D(15,3, data_format="channels_last",activation='relu', padding="same")(conv15_3_1)
    # conv15_3 = Conv2D(15,3, data_format="channels_last",padding="same")(conv15_3_3)
    # reshaped = Reshape((jetDim,jetDim,trackNum,parNum+1))(conv15_3)
    #
    # if(not noProb) :
    #     conv1_3_1 = Conv2D(3,3, data_format="channels_last", activation='sigmoid', padding="same")(conv15_5)
    #     reshaped_prob = Reshape((jetDim,jetDim,trackNum))(conv1_3_1)





    if(not noProb) :
        model = Model([NNinputs,NNinputs_jeta,NNinputs_jpt],[reshaped,reshaped_prob])
    else:
        model = Model([NNinputs,NNinputs_jeta,NNinputs_jpt],reshaped)

    # anubi = keras.optimizers.Adam(lr=0.0001)
    anubi = keras.optimizers.Adam(lr=0.0001)#5 (0.1 su working drop nopt eta) (0.2 su current training per il clipped subito) 0.5 su 200flitri

    # anubi = keras.optimizers.Adam(amsgrad=True)


    # model.compile(optimizer=anubi, loss=['mse','binary_crossentropy'], loss_weights=[1,1]) #0.01,100
    # model.compile(optimizer='adam', loss=['mse',loss_weighted_crossentropy], loss_weights=[1,1]) #0.01,100

    # model.compile(optimizer='adam', loss=[loss_mse_weight(target_loss_w),loss_weighted_crossentropy], loss_weights=[1,1])


    #model.compile(optimizer='adam', loss=[loss_mse_select,loss_weighted_crossentropy], loss_weights=[1,1])
    if(not noProb) :
        if(standard_mse) :
            model.compile(optimizer=anubi, loss=['mse','binary_crossentropy'], loss_weights=[1,1])
        elif (ROI_loss):
            model.compile(optimizer=anubi, loss=[loss_mse_select_clipped,loss_ROI_crossentropy], loss_weights=[1,1])
        else :
            model.compile(optimizer=anubi, loss=[loss_mse_select_clipped,loss_weighted_crossentropy], loss_weights=[1,1])
    else :
        if(standard_mse) :
            model.compile(optimizer=anubi, loss='mse')
        else :
            model.compile(optimizer=anubi, loss=loss_mse_select)
    # model.compile(optimizer='adam', loss=[loss_mse_select,'binary_crossentropy'], loss_weights=[1,1])



    model.summary()


# model = Sequential([
#         Dense(200, activation = 'relu', input_shape=(layNum,jetDim,jetDim) ),
#         Flatten(),
#         Dropout(0.2),
#         Dense(4)
#     ])

#-----------------------------------------NN TRAINING and PREDICITION -----------------------------------#
combined_training = False

#1hit
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/histo_10k_1hitPar*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/histo_10k_1hitPar*.root')

#4hit
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/histo_10_*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/histo_10_*.root')

#4hit 100k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/_100k/histo_50_1hitPar_1*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/_100k/histo_50_1hitPar_40*.root')

#4hit standard mse 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/noflag_4hit/histo_10k_*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/noflag_4hit/histo_10k_*.root')

#1hit standard mse 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/noflag_1hit/histo_10k_*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/noflag_1hit/histo_10k_*.root')

#4hit standard mse 100k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/noflag_100k_4hit/histo_100k_*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/noflag_100k_4hit/histo_100k_*.root')

#4hit multiplied (regularized par) 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/multiplied_4hit/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/multiplied_4hit/histo*.root')

#1hit multiplied (regularized par) 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/multiplied_1hit/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/multiplied_1hit/histo*.root')

#1hit multiplied (regularized par) 10k input regolarize
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/multiplied_1hit_input/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/multiplied_1hit_input/histo*.root')

#1hit multiplied small windows (100), 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data//multiplied_smallWin_1hit_input/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation//multiplied_smallWin_1hit_input/histo*.root')

#1hit multiplied small windows (100), 100k
# files=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/data/multiplied_1hit_100k_smallWin/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNPixSeed/_newWorkCUDA/validation/multiplied_1hit_100k_smallWin/histo*.root')

#1hit multiplied 30x30 CLUSTER CENTER BUGGED
# files=glob.glob('home/users/bertacch/trackjet/NNClustSeed/data/multiplied_1hit/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/validation/multiplied_1hit/histo*.root')

#4hit multiplied 30x30 CLUSTER CENTER 10k
# files=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/data/10k_multiplied_4hit_30/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/validation/10k_multiplied_4hit_30/histo*.root')

#4hit multiplied 30x30 CLUSTER CENTER 100k
# files=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/data/100k_multiplied_4hit_30/histo*.root')
# files_validation=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/validation/100k_multiplied_4hit_30/histo*.root')

#4hit multiplied 30x30 CLUSTER CENTER 1M
files=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/data/1M_multiplied_4hit_30/histo*.root')
files_validation=glob.glob('/home/users/bertacch/trackjet/NNClustSeed/validation/1M_multiplied_4hit_30/histo*.root')

print("number of  file=", len(files))
print("number of file validation=", len(files_validation))

import uproot
tot_events = 0
tot_events_validation = 0
if(uproot_flag) :
    tfile = uproot.open(input_name)
    tree = tfile["demo"]["NNClustSeedInputTree"]
    input_jeta = tree.array("jet_eta")
    tot_events = len(input_jeta)
    tot_events_validation=tot_events*valSplit
    tot_events=tot_events*(1-valSplit)
else :
    for f in files :
        tfile = uproot.open(f)
        tree = tfile["demo"]["NNClustSeedInputTree"]
        input_jeta = tree.array("jet_eta")
        tot_events = tot_events+len(input_jeta)
    for f in files_validation :
        tfile = uproot.open(f)
        tree = tfile["demo"]["NNClustSeedInputTree"]
        input_jeta = tree.array("jet_eta")
        tot_events_validation = tot_events_validation+len(input_jeta)

jetNum = tot_events
jetNum_validation = tot_events_validation
print("total number of events =", jetNum)
print("total number of events validation=", jetNum_validation)

checkpointer = ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5",verbose=1, save_weights_only=True)

if train :
    stepNum = jetNum/batch_size



    if continue_training :
        #model.load_weights('NNPixSeed_train_event_{ev}.h5'.format(ev=jetNum))
        # model.load_weights('weights.61-102.27.hdf5')
        # model.load_weights('NNPixSeed_train_event_16460_tris.h5')
        # model.load_weights('weights.156-63.70.hdf5')
        # model.load_weights('NNPixSeed_train_event_16460_quar.h5')
        # model.load_weights('weights.06-499.39.hdf5')


        # model.load_weights('NNPixSeed_train_event_16460_quin.h5') #working 10k from new_sample_test folder


        # model.load_weights('NNPixSeed_train_event_24922.h5') NUMERO 1
        # model.load_weights('weights.28-7.88.hdf5') #NUMERO2
        # model.load_weights('weights.10-7.80.hdf5') #NEWNUM1
        #model.load_weights('NNPixSeed_train_event_16460_quin.h5')
        # model.load_weights('weights.359-18.62.hdf5')
        # model.load_weights('weights.11-29.39.hdf5')







        # model.load_weights('NNPixSeed_train_event_232072_13.h5')#last final good
        # model.load_weights('weights.155-0.45.hdf5')

        # model.load_weights('NNPixSeed_train_event_232072.h5')
        #model.load_weights('NNPixSeed_train_event_232072_13.h5') #this is the good one

        #model.load_weights('weights.166-0.64.hdf5')
        model.load_weights('NNPixSeed_train_event_1106439_17.h5')


        if(not noProb) :
            if(uproot_flag) :
                history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, epochs=epochs+start_epoch, verbose = 2, validation_split=valSplit,  initial_epoch=start_epoch, callbacks=[checkpointer])#class_weight={'reshape_2':{},'reshape_3':{0:1,1:2000}})  #, callbacks=[validationCall()])
            else :
                history  = model.fit_generator(generator=Generator(files),steps_per_epoch=stepNum, epochs=epochs+start_epoch, verbose = 2, max_queue_size=1, validation_data=Generator(files_validation),  validation_steps=jetNum_validation/batch_size, callbacks=[checkpointer], initial_epoch=start_epoch)
        else :
            if(uproot_flag) :
                history  = model.fit([input_,input_jeta,input_jpt], target_,  batch_size=batch_size, epochs=epochs+start_epoch, verbose = 2, validation_split=valSplit,  initial_epoch=start_epoch, callbacks=[checkpointer])#class_weight={'reshape_2':{},'reshape_3':{0:1,1:2000}})  #, callbacks=[validationCall()])
            else :
                history  = model.fit_generator(generator=Generator(files),steps_per_epoch=stepNum, epochs=epochs+start_epoch, verbose = 2, max_queue_size=1, validation_data=Generator(files_validation),  validation_steps=jetNum_validation/batch_size, callbacks=[checkpointer], initial_epoch=start_epoch)

        model.save_weights('NNPixSeed_train_event_{ev}_23.h5'.format(ev=jetNum))
    # elif combined_training :
    #     # model.load_weights('toyNN_train_bis_17_lay2_comp.h5')
    #     model.load_weights('toyNN_train_COMB_8_lay2_comp.h5')
    #     history  = model.fit(input_, [target_,target_prob],  batch_size=batch_size, nb_epoch=230+epochs, verbose = 2, validation_split=valSplit, initial_epoch=230+1,  callbacks=[validationCall()])
    #     model.save_weights('toyNN_train_COMB_{Seed}_bis_lay2_comp.h5'.format(Seed=seed))
    else :
        # pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_{Seed}_ep{Epoch}.pdf".format(Seed=seed, Epoch=epochs))
        # history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, nb_epoch=epochs, verbose = 2, validation_split=valSplit,  callbacks=[checkpointer])#,class_weight={'reshape_2':{},'reshape_3':{0:1,1:2000}})  #, callbacks=[validationCall()])
        print("Number of Steps=",stepNum)

        if(not noProb) :
            if(uproot_flag and (not Deb1ev)) :
                history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, epochs=epochs, verbose = 2, validation_split=valSplit,  callbacks=[checkpointer])
            elif(Deb1ev) :
                history  = model.fit([input_,input_jeta,input_jpt], [target_,target_prob],  batch_size=batch_size, epochs=epochs, verbose = 2)#, validation_split=valSplit,  callbacks=[checkpointer])
            else :
                history  = model.fit_generator(generator=Generator(files),steps_per_epoch=stepNum, epochs=epochs, verbose = 2, max_queue_size=1, validation_data=Generator(files_validation),  validation_steps=jetNum_validation/batch_size, callbacks=[checkpointer])
        else :
            if(uproot_flag and (not Deb1ev)) :
                history  = model.fit([input_,input_jeta,input_jpt], target_,  batch_size=batch_size, epochs=epochs, verbose = 2, validation_split=valSplit,  callbacks=[checkpointer])
            elif(Deb1ev) :
                history  = model.fit([input_,input_jeta,input_jpt], target_,  batch_size=batch_size, epochs=epochs, verbose = 2)#, validation_split=valSplit,  callbacks=[checkpointer])
            else :
                history  = model.fit_generator(generator=Generator(files),steps_per_epoch=stepNum, epochs=epochs, verbose = 2, max_queue_size=1, validation_data=Generator(files_validation),  validation_steps=jetNum_validation/batch_size, callbacks=[checkpointer])

        # pdf_par.close()
        model.save_weights('NNPixSeed_train_event_{ev}.h5'.format(ev=jetNum))


    pdf_loss = mpl.backends.backend_pdf.PdfPages("loss_file_ep{Epoch}_event{ev}.pdf".format( Epoch=epochs,ev=jetNum))

    if(not noProb) :
        if(not Deb1ev) :
            plt.figure(1000)
            # plt.yscale('log')
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
            # plt.yscale('log')
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
            # plt.yscale('log')
            pylab.plot(history.history['reshape_3_loss'])
            pylab.plot(history.history['val_reshape_3_loss'])
            pylab.title('model loss (probability)')
            pylab.ylabel('loss')
            pylab.xlabel('epoch')
            plt.grid(True)
            pylab.legend(['train', 'test'], loc='upper right')
            # pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
            pdf_loss.savefig(1002)
    else :
        if(not Deb1ev) :
            plt.figure(1000)
            # plt.yscale('log')
            pylab.plot(history.history['loss'])
            pylab.plot(history.history['val_loss'])
            pylab.title('model loss')
            pylab.ylabel('loss')
            pylab.xlabel('epoch')
            plt.grid(True)
            pylab.legend(['train', 'test'], loc='upper right')
            #pylab.savefig("loss_{Seed}.pdf".format(Seed=seed))
            pdf_loss.savefig(1000)


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
        #model.load_weights('weights.24-0.10.hdf5')#good  one
        # model.load_weights('NNPixSeed_train_event_16460_quin.h5')
        # model.load_weights('NNPixSeed_train_event_24922.h5'.format(ev=jetNum))
        # model.load_weights('NNPixSeed_train_event_1106439_17.h5')
        # model.load_weights('NNPixSeed_train_event_24922.h5')
        # model.load_weights('toyNN_train_COMB_8_bis_lay2_comp.h5')
        #model.load_weights('NNPixSeed_train_event_{ev}_bis.h5'.format(ev=jetNum))
        # model.load_weights('../new_deltaphi/NNPixSeed_train_event_1000.h5')

        #model.load_weights('weights.166-0.64.hdf5')
        model.load_weights('NNPixSeed_train_event_1106439_17.h5')#poster one

    if(not noProb) :
        [validation_par,validation_prob] = model.predict([input_,input_jeta,input_jpt])
        validation_par = np.float64(validation_par)
        np.savez("NNPixSeed_prediction_event_{ev}".format(ev=jetNum), validation_par=validation_par, validation_prob=validation_prob)

    else :
        validation_par = model.predict([input_,input_jeta,input_jpt])
        validation_par = np.float64(validation_par)
        np.savez("NNPixSeed_prediction_event_{ev}".format(ev=jetNum), validation_par=validation_par)


# debug the loss-------------------------------------#
    #evaluated_loss =  loss_mse_select(target_,validation_par)
    # byhandloss =0
    # nnn=0;
    # print("N of events=",len(target_))
    # for ev in range(len(target_)) :
    #     for x in range(jetDim) :
    #         for y in range(jetDim) :
    #             for trk in range (trackNum) :
    #                     if(target_[ev][x][y][trk][4]!=0) :
    #                         for par in range(parNum) :
    #                             # print("x=",x,"y=",y,"trk=",trk,"par=",par,"valore=",target_[ev][x][y][trk][par], ", prediction=",validation_par[ev][x][y][trk][par])
    #                             # byhandloss = byhandloss+(target_[ev][x][y][trk][par]-validation_par[ev][x][y][trk][par])**2
    #                             byhandloss = byhandloss+(target_[ev][x][y][trk][par])
    #
    #                             nnn=nnn+1
    # # print("TARGET=", target_)
    # byhandloss=byhandloss/nnn
    # print("number=",nnn)
    # # print("validation_par=", validation_par)
    # print(" LOSS:", evaluated_loss)
    # print("EVALUATED LOSS:", K.eval(evaluated_loss))
    # print("calculated by hand LOSS:", byhandloss)



    print("prediction: completed ")




#------------------------------------------------ PRINT ROOT FILE -----------------------------------#

if output :
     if predict == False :

        print("prediction loading: start")
        loadpred = np.load("NNPixSeed_prediction_event_106.4.npz".format(ev=jetNum))

        validation_par = loadpred['validation_par']
        if(not noProb) :
            validation_prob = loadpred['validation_prob']

        print("prediction loading: completed")

     if(ROI_loss) :
         target_prob = target_prob[:,:,:,:,:-1]
         validation_prob = validation_prob[:,:,:,:,:-1]

     from ROOT import *
     from ROOT import gStyle
     gROOT.Reset()
     gROOT.SetBatch(True); #no draw at screen
     gStyle.SetOptStat(0);#no stats
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
            canvasProb_jet.append(TCanvas("canvasProb_%d_%d" % (jet,trk), "canvasProb_%d_%d" % (jet,trk),500,800))
            mapProbPredTot_jet.append(TH2F("mapProbPredTot_%d_%d" % (jet,trk), "mapProbPredTot_%d_%d" % (jet,trk), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
            # print(len(input_))

         for lay in range(layNum) :

             mapTot_jet.append(TH2F("mapTot_%d_%d" % (jet, lay), "mapTot_%d_%d" % (jet, lay), jetDim,-jetDim/2,jetDim/2,jetDim,-jetDim/2,jetDim/2))
             canvasTot_jet.append(TCanvas("canvasTot_%d_%d" % (jet, lay), "canvasTot_%d_%d" % (jet, lay),500,800))
             graphTargetTot_jet.append(TGraph())
            #  graphPredTot_jet.append(TGraph(trackNum*3))
             graphPredTot_jet.append(TGraph())


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
             graphPredTot[jet][lay].SetMarkerColor(7)
             graphPredTot[jet][lay].SetMarkerStyle(28)
             graphPredTot[jet][lay].SetMarkerSize(3)
             graphTargetTot[jet][lay].SetMarkerColor(6)
             graphTargetTot[jet][lay].SetMarkerStyle(2)
             graphTargetTot[jet][lay].SetMarkerSize(3)
             for x in range(jetDim) :
                 for y in range(jetDim) :
                     mapTot[jet][lay].SetBinContent(x+1,y+1,input_[j_eff][x][y][lay])
                     for trk in range(trackNum) :
                            #  if(trk>0 and target_prob[j_eff][x][y][trk] == 1) :
                            #     print("Secondary map filled: map, x,y,jet",trk,x,y,jet)
                            #  print("jet,trk,x,y,j_eff",jet,trk,x,y,j_eff)
                        if(not noProb) :
                             mapProbPredTot[jet][trk].SetBinContent(x+1,y+1,validation_prob[j_eff][x][y][trk])
                             if target_prob[j_eff][x][y][trk] == 1 and lay==1:
                                 xx= float(target_[j_eff][x][y][trk][0])/float(0.01)*0.01#normaliz. factor
                                 yy= float(target_[j_eff][x][y][trk][1])/float(0.015)*0.01
                                 graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                 tarPoint = tarPoint+1

                                 x0,y0 = prop_on_layer(x+xx-jetDim/2, y+yy-jetDim/2,target_[j_eff][x][y][trk][2]*0.01,target_[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],0)
                                 x2,y2 = prop_on_layer(x+xx-jetDim/2, y+yy-jetDim/2,target_[j_eff][x][y][trk][2]*0.01,target_[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],2)
                                 x3,y3 = prop_on_layer(x+xx-jetDim/2, y+yy-jetDim/2,target_[j_eff][x][y][trk][2]*0.01,target_[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],3)
                                 graphTargetTot[jet][0].SetPoint(tarPoint,x0,y0)
                                 graphTargetTot[jet][2].SetPoint(tarPoint,x2,y2)
                                 graphTargetTot[jet][3].SetPoint(tarPoint,x3,y3)
                                #  print(" x0,y0=",x0,y0, ",x1,y1=",x+xx-jetDim/2, y+yy-jetDim/2, "x2,y2=",x2,y2," x3,y3=",x3,y3,)
                                #  print("-----------------------------------------------------")

                                #  latex_target = TLatex(graphTargetTot[jet][lay].GetX()[tarPoint-1], graphTargetTot[jet][lay].GetY()[tarPoint-1],"%.3f, %.3f"%(target_[j_eff][x][y][trk][2],target_[j_eff][x][y][trk][3]));
                                #  latex_target.SetTextSize(0.02)
                                #  latex_target.SetTextColor(2)
                                #  graphTargetTot[jet][lay].GetListOfFunctions().Add(latex_target);

                            #  if(target_[j_eff][x][y][trk][0]>0.000001 or target_[j_eff][x][y][trk][0]<- 0.00001 or target_[j_eff][x][y][trk][0]>0.000001 or target_[j_eff][x][y][trk][0]<- 0.00001 or target_[j_eff][x][y][trk][1]>0.000001 or target_[j_eff][x][y][trk][1]<- 0.00001 or target_[j_eff][x][y][trk][2]>0.000001 and target_[j_eff][x][y][trk][2]<- 0.00001 or target_[j_eff][x][y][trk][3]>0.000001 and target_[j_eff][x][y][trk][3]<- 0.00001) :
                            #      if(lay==1) :
                            #             xx= target_[j_eff][x][y][trk][0]/0.01+0.01*x
                            #             yy= target_[j_eff][x][y][trk][1]/0.015+0.01*y
                            #             graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                            #             tarPoint = tarPoint+1
                            #             print("________________________________________")
                            #             print("New not null, bin (x,y):",x,y)
                            #             print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3], "Probabiity target=", target_prob[j_eff][x][y][trk])
                             if validation_prob[j_eff][x][y][trk] > prob_thr and lay==1 : #and   target_prob[j_eff][x][y][trk] == 1: #QUESTA E' la COSA GIUSTA SE NON DEBUGGO
                             #if target_[j_eff][x][y][trk][4]!=0 and lay==1:
                             #if target_prob[j_eff][x][y][trk]!=0 and lay==1:
                                 xx_pr= float(validation_par[j_eff][x][y][trk][0])/float(0.01)*0.01
                                 yy_pr= float(validation_par[j_eff][x][y][trk][1])/float(0.015)*0.01
                                 graphPredTot[jet][lay].SetPoint(predPoint,x+xx_pr-jetDim/2,y+yy_pr-jetDim/2)

                                 x0,y0 = prop_on_layer(x+xx_pr-jetDim/2, y+yy_pr-jetDim/2,validation_par[j_eff][x][y][trk][2]*0.01,validation_par[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],0)
                                 x2,y2 = prop_on_layer(x+xx_pr-jetDim/2, y+yy_pr-jetDim/2,validation_par[j_eff][x][y][trk][2]*0.01,validation_par[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],2)
                                 x3,y3 = prop_on_layer(x+xx_pr-jetDim/2, y+yy_pr-jetDim/2,validation_par[j_eff][x][y][trk][2]*0.01,validation_par[j_eff][x][y][trk][3]*0.01,input_jeta[j_eff],3)
                                 graphPredTot[jet][0].SetPoint(tarPoint,x0,y0)
                                 graphPredTot[jet][2].SetPoint(tarPoint,x2,y2)
                                 graphPredTot[jet][3].SetPoint(tarPoint,x3,y3)



                                 predPoint = predPoint+1
                                #  latex_pred = TLatex(graphPredTot[jet][lay].GetX()[predPoint-1], graphPredTot[jet][lay].GetY()[predPoint-1],"%.3f, %.3f"%(validation_par[j_eff][x][y][trk][2],validation_par[j_eff][x][y][trk][3]));
                                #  latex_pred.SetTextSize(0.02)
                                #  graphPredTot[jet][lay].GetListOfFunctions().Add(latex_pred);

                                 print("________________________________________")
                                 print("New Pred, bin (x,y):",x-jetDim/2,y-jetDim/2)
                                #  print("Flag=",target_[j_eff][x][y][trk][4], "track=", trk)
                                 print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3], "Probabiity target=", target_prob[j_eff][x][y][trk])
                                 print("prediction(x,y,eta,phi)=",validation_par[j_eff][x][y][trk][0]," ", validation_par[j_eff][x][y][trk][1]," ",validation_par[j_eff][x][y][trk][2]," ",validation_par[j_eff][x][y][trk][3], "Probabiity pred=", validation_prob[j_eff][x][y][trk])
                                 print(" x0,y0=",x0,y0," x2,y2=",x2,y2," x3,y3=",x3,y3,)
                            #  if(target_[j_eff][x][y][trk][0]!=0.0 or target_[j_eff][x][y][trk][1]!=0.0 or target_[j_eff][x][y][trk][2]!=0.0 or target_[j_eff][x][y][trk][3]!=0.0 ) :
                                #   print("---------------")
                                #   print("New Not-null-Target, bin (x,y):",x,y)
                                #   print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3], "Probabiity target=", target_prob[j_eff][x][y][trk])
                                #   print("prediction(x,y,eta,phi)=",validation_par[j_eff][x][y][trk][0]," ", validation_par[j_eff][x][y][trk][1]," ",validation_par[j_eff][x][y][trk][2]," ",validation_par[j_eff][x][y][trk][3])
                        else :
                             if target_[j_eff][x][y][trk][4] != 0 and lay==1:
                                 xx= float(target_[j_eff][x][y][trk][0])/float(0.01)
                                 yy= float(target_[j_eff][x][y][trk][1])/float(0.015)
                                 graphTargetTot[jet][lay].SetPoint(tarPoint,x+xx-jetDim/2,y+yy-jetDim/2)
                                 tarPoint = tarPoint+1
                             if target_[j_eff][x][y][trk][4]!=0 and lay==1:
                                 xx_pr= float(validation_par[j_eff][x][y][trk][0])/float(0.01)*0.01
                                 yy_pr= float(validation_par[j_eff][x][y][trk][1])/float(0.015)*0.01

                                 graphPredTot[jet][lay].SetPoint(predPoint,x+xx_pr-jetDim/2,y+yy_pr-jetDim/2)
                                 predPoint = predPoint+1
                                 print("________________________________________")
                                 print("New Pred, bin (x,y):",x-jetDim/2,y-jetDim/2)
                                #  print("Flag=",target_[j_eff][x][y][trk][4], "track=", trk)
                                 print("target(x,y,eta,phi)=",target_[j_eff][x][y][trk][0]," ", target_[j_eff][x][y][trk][1]," ",target_[j_eff][x][y][trk][2]," ",target_[j_eff][x][y][trk][3])
                                 print("prediction(x,y,eta,phi)=",validation_par[j_eff][x][y][trk][0]," ", validation_par[j_eff][x][y][trk][1]," ",validation_par[j_eff][x][y][trk][2]," ",validation_par[j_eff][x][y][trk][3])



     RGB=False
     output_file = TFile("NNPixSeed_mapValidation_events_{ev}.root".format(ev=jetNum),"recreate")
     from array import array as array2
    #  from ROOT import gStyle, TColor
    #  Red =[1.,1.]
    #  Green =[1.,0.]
    #  Blue =[1.,0.]
    #  Stops =[0.,1.]
    #  StopsArray = array2('d', Stops)
    #  RedArray = array2('d', Red)
    #  GreenArray = array2('d', Green)
    #  BlueArray = array2('d', Blue)
    #  TColor.CreateGradientColorTable(2, StopsArray, RedArray, GreenArray, BlueArray, 20)

     if(RGB) :

         NCont=10

         array_of_palette = []
         palette = []

         Red =[1.,1.]
         Green =[1.,0.]
         Blue =[1.,0.]
         Stops =[0.,1.]
         StopsArray = array2('d', Stops)
         RedArray = array2('d', Red)
         GreenArray = array2('d', Green)
         BlueArray = array2('d', Blue)
         FI = TColor.CreateGradientColorTable(2, StopsArray, RedArray, GreenArray, BlueArray, NCont)
         for i in range(0,NCont) :
             palette.append(FI+i)
         paletteArray = array2('i',palette)
         palette[:]=[]
         array_of_palette.append(paletteArray)


         Red =[1.,0.]
         Green =[1.,0.]
         Blue =[1.,0.]
         Stops =[0.,1.]
         StopsArray = array2('d', Stops)
         RedArray = array2('d', Red)
         GreenArray = array2('d', Green)
         BlueArray = array2('d', Blue)
         FI = TColor.CreateGradientColorTable(2, StopsArray, RedArray, GreenArray, BlueArray, NCont)
         for i in range(0,NCont) :
             palette.append(FI+i)
         paletteArray = array2('i',palette)
         palette[:]=[]
         array_of_palette.append(paletteArray)

         Red =[1.,0]#[1.,1] #(yellow)
         Green =[1.,1]
         Blue =[1.,0.]
         Stops =[0.,1.]
         StopsArray = array2('d', Stops)
         RedArray = array2('d', Red)
         GreenArray = array2('d', Green)
         BlueArray = array2('d', Blue)
         FI = TColor.CreateGradientColorTable(2, StopsArray, RedArray, GreenArray, BlueArray, NCont)
         for i in range(0,NCont) :
             palette.append(FI+i)
         paletteArray = array2('i',palette)
         palette[:]=[]
         array_of_palette.append(paletteArray)

         Red =[1.,0.]
         Green =[1.,0.]
         Blue =[1.,1.]
         Stops =[0.,1.]
         StopsArray = array2('d', Stops)
         RedArray = array2('d', Red)
         GreenArray = array2('d', Green)
         BlueArray = array2('d', Blue)
         FI = TColor.CreateGradientColorTable(2, StopsArray, RedArray, GreenArray, BlueArray, NCont)
         for i in range(0,NCont) :
             palette.append(FI+i)
         paletteArray = array2('i',palette)
         palette[:]=[]
         array_of_palette.append(paletteArray)


     for jet in range(numPrint) :
         for lay in range(layNum) :
             canvasTot[jet][lay].cd()
             mapTot[jet][lay].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].GetYaxis().SetRangeUser(-jetDim,jetDim)
             mapTot[jet][lay].SetTitle("Pixel Map, cluster %d, layer %d" % (jet, lay+1))
             mapTot[jet][lay].GetXaxis().SetTitle("x [pixel]")
             mapTot[jet][lay].GetYaxis().SetTitle("y [pixel]")
             mapTot[jet][lay].GetYaxis().SetTitleOffset(1)
             mapTot[jet][lay].GetZaxis().SetTitle("ADC count [/14k]")
             mapTot[jet][lay].GetZaxis().SetTitleOffset(-1.05)
             mapTot[jet][lay].GetXaxis().SetTitleSize(0.06)
             mapTot[jet][lay].GetYaxis().SetTitleSize(0.06)
             mapTot[jet][lay].GetZaxis().SetTitleSize(0.04)
             mapTot[jet][lay].GetXaxis().SetTitleOffset(0.7)
             mapTot[jet][lay].GetYaxis().SetTitleOffset(0.6)

             if(not RGB) :
                 mapTot[jet][lay].Draw("colz")
             else :
                 gStyle.SetPalette(NCont,array_of_palette[lay])
                 mapTot[jet][lay].Draw("colz")
             if(jet==outEvent and RGB):
                 canvasTot[jet][lay].SaveAs("RGB_PixelMap_input_layer%d_event%d.pdf" % (lay,jet))

             graphTargetTot[jet][lay].Draw("SAME P")
             graphPredTot[jet][lay].Draw("SAME P")

             graphTargetTot[jet][lay].SetLineColor(0)
             graphPredTot[jet][lay].SetLineColor(0)
             graphTargetTot[jet][lay].SetFillColor(0)
             graphPredTot[jet][lay].SetFillColor(0)

             legTitle = TLegend(0.1,1,0.7,0.85);
             legTitle.SetHeader("CMS Private")
             legTitle.SetBorderSize(0)
             legTitle.Draw("SAME")
             legTitle.SetFillStyle(0);


             legTot = TLegend(0.1,0.9,0.3,0.8);
            #  legTot.SetHeader("CMS Private")
            #  legTot.AddEntry(mapTot[jet][lay], "Pixel ADC counts")
             legTot.AddEntry(graphTargetTot[jet][lay], "Target")
             legTot.AddEntry(graphPredTot[jet][lay], "Prediction")
             legTot.SetTextSize(0.03);
             legTot.Draw("SAME")

             canvasTot[jet][lay].Write()

             if(jet==outEvent and RGB):
                 canvasTot[jet][lay].SaveAs("RGB_PixelMap_crosses_layer%d_event%d.pdf" % (lay,jet))#.png

         for trk in range(trackNum) :
             canvasProb[jet][trk].cd()
             mapProbPredTot[jet][trk].GetXaxis().SetRangeUser(-jetDim,jetDim)
             mapProbPredTot[jet][trk].GetYaxis().SetRangeUser(-jetDim,jetDim)
             gStyle.SetPalette(57)
             mapProbPredTot[jet][trk].Draw("colz")
             mapProbPredTot[jet][trk].GetXaxis().SetTitle("x [pixel]")
             mapProbPredTot[jet][trk].GetYaxis().SetTitle("y [pixel]")
             mapProbPredTot[jet][trk].GetYaxis().SetTitleOffset(1.2)
             mapProbPredTot[jet][trk].SetTitle("TCP Prediction Map, cluster %d, overlap %d" % (jet, trk))
             mapProbPredTot[jet][trk].GetZaxis().SetTitle("Probability")
             mapProbPredTot[jet][trk].GetZaxis().SetTitleOffset(-1.05)
             mapProbPredTot[jet][trk].GetXaxis().SetTitleSize(0.06)
             mapProbPredTot[jet][trk].GetYaxis().SetTitleSize(0.06)
             mapProbPredTot[jet][trk].GetXaxis().SetTitleOffset(0.7)
             mapProbPredTot[jet][trk].GetYaxis().SetTitleOffset(0.6)
             graphTargetTot[jet][1].Draw("SAME P")
             graphPredTot[jet][1].Draw("SAME P")

             legTitleProb = TLegend(0.1,1,0.7,0.85);
             legTitleProb.SetHeader("CMS Private")
             legTitleProb.SetBorderSize(0)
             legTitleProb.Draw("SAME")
             legTitleProb.SetFillStyle(0);


             legProb = TLegend(0.1,0.9,0.3,0.8);
             legProb.AddEntry(graphTargetTot[jet][1], "Target")
             legProb.AddEntry(graphPredTot[jet][1], "Prediction")
             legProb.SetTextSize(0.03);
             legProb.Draw("SAME")


             canvasProb[jet][trk].Write()

         if(jet==outEvent and RGB):
             canvasProb[jet][0].SaveAs("Probabiltiy_crosses_event%d.pdf" % (jet))#.png

     output_file.Close()





     aconical_sel = False
     if(not noProb) :
         print("parameter file: start looping")
         pdf_par = mpl.backends.backend_pdf.PdfPages("parameter_file_events_{ev}.pdf".format(ev=jetNum))

         for par in range(parNum) :
             bins = []# np.zeros(shape=(int(jetNum*valSplit)))
             bins_pred = []
             bins_target = []
             nbin =0
             n_sig_ok = 0
            #  for j in range (int(len(input_)*valSplit)) :
            #      j_eff = j+int(len(input_)*(1-valSplit))
             for j in range (int(len(input_))) :
                 j_eff = j#j+int(len(input_)*(1-valSplit))
                 for x in range(jetDim) :
                     for y in range(jetDim) :
                         for trk in range(trackNum) :
                            #  if validation_prob[j_eff][x][y][trk] > prob_thr :
                             if target_prob[j_eff][x][y][trk] == 1 :
                                 aconical = True
                                 if(x>jetDim/2 and y>jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]<0 and target_[j_eff][x][y][trk][3]<0) :
                                         aconical = False
                                 if(x<=jetDim/2 and y<jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]>0 and target_[j_eff][x][y][trk][3]>0) :
                                         aconical = False
                                 if(x>jetDim/2 and y<=jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]>0 and target_[j_eff][x][y][trk][3]<0) :
                                         aconical = False
                                 if(x<=jetDim/2 and y>jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]<0 and target_[j_eff][x][y][trk][3]>0) :
                                         aconical = False
                                 if(not aconical_sel) :
                                        aconical= True
                                 if(aconical) :
                            #  if validation_prob[j_eff][x][y][trk] > prob_thr :# and target_prob[j_eff][x][y][trk] == 1:
                                # if validation_par[j_eff][x][y][trk][0] != 0 or validation_par[j_eff][x][y][trk][1] != 0  or validation_par[j_eff][x][y][trk][2] != 0 or validation_par[j_eff][x][y][3] != 0 :
                                     bins.append((validation_par[j_eff][x][y][trk][par] - target_[j_eff][x][y][trk][par])*0.01)
                                     bins_pred.append(validation_par[j_eff][x][y][trk][par]*0.01)
                                     bins_target.append(target_[j_eff][x][y][trk][par]*0.01)
                                     nbin = nbin+1
                                     if(validation_par[j_eff][x][y][trk][par]*target_[j_eff][x][y][trk][par]>0) : #same Sign
                                        n_sig_ok = n_sig_ok+1
             fracsig=n_sig_ok/float(nbin)
             print("Parameter {PAR}, number of correct sign={n}, fraction={f}".format(PAR=par, n=n_sig_ok, f=fracsig))
             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins,70, facecolor='darkorange', alpha=0.75, range=(-0.03,0.03))
             if(par == 0) :
                 pylab.title('Residual distribution - x',fontsize=24)
             if(par == 1) :
                 pylab.title('Residual distribution - y',fontsize=24)
             if(par == 2) :
                 pylab.title('Residual distribution - $\eta$',fontsize=24)
             if(par == 3) :
                 pylab.title('Residual distribution - $\phi$',fontsize=24)

            #  pylab.title('Residual distribution_par{PAR}'.format(PAR=par),fontsize=24)
             pylab.ylabel('entries',fontsize=22)
             if(par==0 or par==1) :
                 pylab.xlabel('prediction-target [cm]',fontsize=22)
             else :
                 pylab.xlabel('prediction-target',fontsize=22)

            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             plt.grid(True)
             if(RGB) :
                 pylab.savefig("residual_{jj}_{PAR}.pdf".format(PAR=par,jj=jetNum))#.png
             pdf_par.savefig()

             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins_target,70, facecolor='royalblue', alpha=0.75, range=(-0.03,0.03))
            #  pylab.title('Target distribution_par{PAR}'.format(PAR=par),fontsize=24)
             if(par == 0) :
                 pylab.title('Target distribution - x',fontsize=24)
             if(par == 1) :
                 pylab.title('Target distribution - y',fontsize=24)
             if(par == 2) :
                 pylab.title('Target distribution - $\eta$',fontsize=24)
             if(par == 3) :
                 pylab.title('Target distribution - $\phi$',fontsize=24)
             if(par==0 or par==1) :
                 pylab.xlabel('target [cm]',fontsize=22)
             else :
                 pylab.xlabel('target',fontsize=22)
             pylab.ylabel('entries',fontsize=22)
            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             plt.grid(True)
             if(RGB) :
                 pylab.savefig("target_{jj}_{PAR}.pdf".format(PAR=par,jj=jetNum))#.png
             pdf_par.savefig()

             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins_pred,70, facecolor='red', alpha=0.75, range=(-0.03,0.03))
            #  pylab.title('Prediction distribution_par{PAR}'.format(PAR=par),fontsize=24)
            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             if(par == 0) :
                 pylab.title('Prediction distribution - x',fontsize=24)
             if(par == 1) :
                 pylab.title('Prediction distribution - y',fontsize=24)
             if(par == 2) :
                 pylab.title('Prediction distribution - $\eta$',fontsize=24)
             if(par == 3) :
                 pylab.title('Prediction distribution - $\phi$',fontsize=24)
             if(par==0 or par==1) :
                 pylab.xlabel('prediction [cm]',fontsize=22)
             else :
                 pylab.xlabel('prediction',fontsize=22)
             pylab.ylabel('entries',fontsize=22)
             plt.grid(True)
             if(RGB) :
                 pylab.savefig("prediction_{jj}_{PAR}.pdf".format(PAR=par,jj=jetNum))#.png
             pdf_par.savefig()



             #now the core!
         for par in range(parNum) :
             bins = []# np.zeros(shape=(int(jetNum*valSplit)))
             bins_pred = []
             bins_target = []
             nbin =0
             n_sig_ok = 0
            #  for j in range (int(len(input_)*valSplit)) :
            #      j_eff = j+int(len(input_)*(1-valSplit))
             for j in range (int(len(input_))) :
                 j_eff = j#j+int(len(input_)*(1-valSplit))
                 for x in range(10,20) :
                     for y in range(10,20) :
                         for trk in range(trackNum) :
                            #  if validation_prob[j_eff][x][y][trk] > prob_thr :
                             if target_prob[j_eff][x][y][trk] == 1 :
                                 aconical = True
                                 if(x>jetDim/2 and y>jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]<0 and target_[j_eff][x][y][trk][3]<0) :
                                         aconical = False
                                 if(x<=jetDim/2 and y<jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]>0 and target_[j_eff][x][y][trk][3]>0) :
                                         aconical = False
                                 if(x>jetDim/2 and y<=jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]>0 and target_[j_eff][x][y][trk][3]<0) :
                                         aconical = False
                                 if(x<=jetDim/2 and y>jetDim/2) :
                                     if(target_[j_eff][x][y][trk][2]<0 and target_[j_eff][x][y][trk][3]>0) :
                                         aconical = False
                                 if(not aconical_sel) :
                                        aconical= True
                                 if(aconical) :
                                 #  if validation_prob[j_eff][x][y][trk] > prob_thr :# and target_prob[j_eff][x][y][trk] == 1:
                                # if validation_par[j_eff][x][y][trk][0] != 0 or validation_par[j_eff][x][y][trk][1] != 0  or validation_par[j_eff][x][y][trk][2] != 0 or validation_par[j_eff][x][y][3] != 0 :
                                     bins.append((validation_par[j_eff][x][y][trk][par] - target_[j_eff][x][y][trk][par])*0.01)
                                     bins_pred.append(validation_par[j_eff][x][y][trk][par]*0.01)
                                     bins_target.append(target_[j_eff][x][y][trk][par]*0.01)
                                     nbin = nbin+1
                                     if(validation_par[j_eff][x][y][trk][par]*target_[j_eff][x][y][trk][par]>0) : #same Sign
                                        n_sig_ok = n_sig_ok+1
             fracsig=n_sig_ok/float(nbin)
             print("Parameter {PAR}, number of correct sign={n}, fraction={f}".format(PAR=par, n=n_sig_ok, f=fracsig))
             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins,100, facecolor='darkorange', alpha=0.75, range=(-0.03,0.03))
             pylab.title('Core - Residual distribution_par{PAR}'.format(PAR=par))
             pylab.ylabel('entries')
             pylab.xlabel('prediction-target')
            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             plt.grid(True)
             # pylab.savefig("parameter_error_{EPOCH}_{PAR}.pdf".format(PAR=par,EPOCH=epoch))
             pdf_par.savefig()

             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins_target,100, facecolor='red', alpha=0.75, range=(-0.03,0.03))
             pylab.title('Core - Target distribution_par{PAR}'.format(PAR=par))
             pylab.ylabel('entries')
             pylab.xlabel('target')
            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             plt.grid(True)
             pdf_par.savefig()

             plt.figure()
            #  plt.yscale('log')
             pylab.hist(bins_pred,100, facecolor='blue', alpha=0.75, range=(-0.03,0.03))
             pylab.title('Core - Prediction distribution_par{PAR}'.format(PAR=par))
             pylab.ylabel('entries')
             pylab.xlabel('prediction')
            #  pylab.xlim(xmax=3)
            #  pylab.xlim(xmin=-3)
             plt.grid(True)
             pdf_par.savefig()

         pdf_par.close()
