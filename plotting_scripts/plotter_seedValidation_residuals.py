import math
from ROOT import *
gROOT.SetBatch(True)
import numpy as np


import math
# from ROOT import *
import ROOT
from ROOT import gStyle
ROOT.gROOT.Reset()
ROOT.gROOT.SetBatch(True)
import numpy as np
import os
import argparse
import array as arr
gStyle.SetOptStat(0)

parser = argparse.ArgumentParser("plotter")
parser.add_argument('-i1',      '--input1',       type=str,                                           action='store', help='input file 1. No default provided')
parser.add_argument('-i2',   '--input2',    type=str,                                           action='store', help='input file 2. No default provided')
parser.add_argument('-i3',    '--input3',     type=str,                                           action='store', help='input file 3. No default provided')
parser.add_argument('-n1',      '--name1',       type=str,     default='deepcore',                     action='store', help='input file 1. deepcore as default')
parser.add_argument('-n2',   '--name2',    type=str,     default='standard',             action='store', help='input file 2.  standard as default')
parser.add_argument('-n3',    '--name3',     type=str,     default='old deepcore',             action='store', help='input file 3. old deepcore as default')
parser.add_argument('-a1',    '--assoc1',     type=str,     default='trackAssociatorByChi2',             action='store', help='associator file 1. by chi2 as default')
parser.add_argument('-a2',    '--assoc2',     type=str,     default='trackAssociatorByChi2',             action='store', help='associator file 1. by chi2 as default')
parser.add_argument('-a3',    '--assoc3',     type=str,     default='trackAssociatorByChi2',             action='store', help='associator file 1. by chi2 as default') #quickAssociatorByHits


args = parser.parse_args()
input1   = args.input1
input2  = args.input2
input3   = args.input3
name1 = args.name1
name2 = args.name2
name3 = args.name3
assoc1 = args.assoc1
assoc2 = args.assoc2
assoc3 = args.assoc3

FWliteAna = False


# fileList = ['deepCore','standard','deepCoreReb']#,'deepCoreBackFit']
# fileList = ['deepCore','standard']

fileList = ['in1','in2','in3']
legLabel = {
   'in1'  : name1,
   'in2'  : name2,
   'in3'  : name3,
}



if FWliteAna :
    parList = ['dxy','dz','phi','ptinv','eta']
else :
    parList = ['dxy','dz','phi','pt','theta']
parListBis = ['Dxy','Dz','Phi','Pt','Theta']

# unitList = ['d_{xy} [cm]','d_{z} [cm]', '#phi', 'p_{T} [GeV]','#theta']
unitList = ['[cm]','[cm]', '', '[GeV]','']

kindList = ['pull','res']

#trackList and pathList must be aligned! 
trackList = ['seed','track']
# pathList = ["DQMData/Run 1/Tracking/Run summary/TrackSeeding/seedjetCoreRegionalStepSeeds_trackAssociatorByChi2/","DQMData/Run 1/Tracking/Run summary/Track/cutsRecoJetCoreRegionalStepByOriginalAlgo_MTVAssociationByChi2/"]
pathList = ["DQMData/Run 1/Tracking/Run summary/JetCore/TrackSeeding/seedjetCoreRegionalStepSeeds_trackAssociatorByChi2/","DQMData/Run 1/Tracking/Run summary/JetCore/cutsRecoJetCoreRegionalStepByOriginalAlgo_trackAssociatorByChi2/"]
# pathList = ["DQMData/Run 1/Tracking/Run summary/JetCore/TrackSeeding/seedjetCoreRegionalStepSeeds_quickAssociatorByHits/","DQMData/Run 1/Tracking/Run summary/JetCore/jetCoreRegionalStep_quickAssociatorByHits/"]


colorDict = {}
colorDict['in1'+'seed'] = 632+2 #red
colorDict['in1'+'track'] = 800-3 #orange
colorDict['in2'+'seed'] = 600+2 #blue
colorDict['in2'+'track'] = 860+6 #azure
colorDict['in3'+'seed'] = 416+2 #dark green
colorDict['in3'+'track'] = 416-4 #light green
# colorDict['deepCoreBackFit'+'seed'] = 880+2 #dark pink
# colorDict['deepCoreBackFit'+'track'] = 880-4 #light pink

def findmax(histo1,histo2) :
    val1 = histo1.GetMaximum()
    val2 = histo2.GetMaximum()
    if val1>val2 : 
        return val1
    else : 
        return val2
        

fileDict = {}

if FWliteAna :
    fileDict['in1'] = TFile.Open("/gpfs/ddn/users/bertacch/cms/CMSSW_10_5_0_pre2/src/test_deepCore_seedValidation/FWlite_analysis/deepCore_norebuild/FWlite_histos_noRebuild_withTP_JCcollection_CMSSWaligned_3.root ")
    fileDict['in2'] = TFile.Open("/gpfs/ddn/users/bertacch/cms/CMSSW_10_5_0_pre2/src/test_deepCore_seedValidation/FWlite_analysis/standard/FWlite_histos_standard_CMSSWaligned_3.root")
    fileDict['in3'] = TFile.Open("/gpfs/ddn/users/bertacch/cms/CMSSW_10_5_0_pre2/src/test_deepCore_seedValidation/FWlite_analysis/deepCore_rebuild/FWlite_histos_withTP_JCcollection_CMSSWaligned_3.root")

else :        
    
    fileDict['in1'] = ROOT.TFile.Open(input1)
    fileDict['in2'] = ROOT.TFile.Open(input2)
    fileDict['in3'] = ROOT.TFile.Open(input3)


histoDict = {}
for f in fileList :
    for par in parList :
        for t in trackList :
            for k in kindList :
                # if k=='pull' :
                #     histoDict[f+par+t+k] = fileDict[f].Get(pathList[trackList.index(t)]+k+parListBis[parList.index(par)])
                #     histoDict[f+par+t+k].GetXaxis().SetTitle(par)  
                #     histoDict[f+par+t+k].SetTitle(f+'_'+t+'_'+k+'_'+par)
                if k=='res' or 1:
                    if par == 'theta' and k=='res': #missing theta for res 2D plots -.-"
                        par2 = 'eta'
                    else :
                        par2 = par
                    if FWliteAna:
                        # if k=='pull' and t=='seed' : #DEBUGGGGG
                        #     histoDict[f+par+t+k] = TH1F()
                        # else :
                        histoDict[f+par+t+k] = fileDict[f].Get(par2+'_'+t+'_'+k)
                    else :
                        # h2_temp = fileDict[f].Get(pathList[trackList.index(t)]+par2+k+'_vs_eta') 
                        pathString = pathList[trackList.index(t)]
                        if f=='in1' : pathStringMod = pathString.replace("trackAssociatorByChi2",assoc1)
                        if f=='in2' : pathStringMod = pathString.replace("trackAssociatorByChi2",assoc2)
                        if f=='in3' : pathStringMod = pathString.replace("trackAssociatorByChi2",assoc3)
                        h2_temp = fileDict[f].Get(pathStringMod+par2+k+'_vs_eta') 

                        histoDict[f+par+t+k] = h2_temp.ProjectionY(f+'_'+t+'_'+k+'_'+par2,0,-1) 
                        # print f, par, t, k, "std dev=", histoDict[f+par+t+k].GetStdDev()
                    # histoDict[f+par+t+k].Rebin(5)
                if k=='res' :
                    histoDict[f+par+t+k].GetXaxis().SetTitle(par2+' '+unitList[parList.index(par)])
                histoDict[f+par+t+k].SetLineWidth(3)
                histoDict[f+par+t+k].SetLineColor(colorDict[f+t])
                histoDict[f+par+t+k].GetYaxis().SetTitle('dN/d'+par+ ' /'+str(histoDict[f+par+t+k].GetBinWidth(1)))
                
                #debug
                # if t=='track' :
                #      histoDict[f+par+t+k].Scale(histoDict[f+par+'seed'+k].Integral()/histoDict[f+par+t+k].Integral())

output= TFile("seedValidation_residuals.root","recreate")

#canvas
canvasDict = {}
# cloneDict = {} #dizionario per plottare gli istogrammi 2 volte con y axis diversi perche root e una merda
for par in parList :
    for k in kindList :
        
        #stanrdard vs deepcore
        for t in trackList :
            canvasDict['leg'+par+k+t+'file'] = ROOT.TLegend(0.7,0.8,0.95,0.95) 
            canvasDict[par+k+t+'file'] = ROOT.TCanvas('c_standard_vs_deepCore_'+par+'_'+k+'_'+t,'c_standard_vs_deepCore_'+par+'_'+k+'_'+t,800,600)
            canvasDict[par+k+t+'file'].cd()
            canvasDict[par+k+t+'file'].SetGridx()
            canvasDict[par+k+t+'file'].SetGridy()
            sameFlag = ''
            maxval = findmax(histo1=histoDict['in2'+par+t+k],histo2=histoDict['in1'+par+t+k])
            for f in fileList :
                histoDict[f+par+t+k].GetYaxis().SetRangeUser(0,maxval+0.01*maxval)
                histoDict[f+par+t+k].DrawCopy(sameFlag)
                # histoDict[f+par+t+k].SetTitle('standard_vs_deepCore_'+par+'_'+k+'_'+t)
                sameFlag = 'SAME'
                # canvasDict['leg'+par+k+t+'file'].AddEntry(histoDict[f+par+t+k], f+', #sigma='+str(histoDict[f+par+t+k].GetStdDev()))
                canvasDict['leg'+par+k+t+'file'].AddEntry(histoDict[f+par+t+k], legLabel[f])#+', #sigma=%.2f'%histoDict[f+par+t+k].GetStdDev())
                # print f, par, t, k, "std dev=", histoDict[f+par+t+k].GetStdDev() 

            canvasDict['leg'+par+k+t+'file'].Draw("SAME")
        
        #seed vs track
        for f in fileList :
            canvasDict['leg'+par+k+f+'track'] = ROOT.TLegend(0.7,0.8,0.95,0.95)
            canvasDict[par+k+f+'track'] = ROOT.TCanvas('c_track_vs_seed_'+par+'_'+k+'_'+legLabel[f]+f,'c_track_vs_seed_'+par+'_'+k+'_'+legLabel[f]+f,800,600)
            canvasDict[par+k+f+'track'].cd()
            canvasDict[par+k+f+'track'].SetGridx()
            canvasDict[par+k+f+'track'].SetGridy()
            sameFlag = ''
            maxval = findmax(histo1=histoDict[f+par+'track'+k],histo2=histoDict[f+par+'seed'+k])
            for t in trackList :
                histoDict[f+par+t+k].GetYaxis().SetRangeUser(0,maxval+0.01*maxval)
                histoDict[f+par+t+k].DrawCopy(sameFlag)
                # histoDict[f+par+t+k].SetTitle('track_vs_seed_'+par+'_'+k+'_'+f)                
                sameFlag = 'SAME'
                canvasDict['leg'+par+k+f+'track'].AddEntry(histoDict[f+par+t+k], t)#+', #sigma=%.2f'%histoDict[f+par+t+k].GetStdDev())
            canvasDict['leg'+par+k+f+'track'].Draw("SAME")

#writing
output.cd()
for f in fileList :
    for par in parList :
        for t in trackList :
            for k in kindList :
                histoDict[f+par+t+k].Write()
for par in parList :
    for k in kindList : 
        for t in trackList :
            canvasDict[par+k+t+'file'].Write()
        for f in fileList :
              canvasDict[par+k+f+'track'].Write()
              