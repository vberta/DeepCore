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
parser.add_argument('-n1',      '--name1',       type=str,     default='deepcore',                     action='store', help='input file 1. No default provided')
parser.add_argument('-n2',   '--name2',    type=str,     default='standard',             action='store', help='input file 2. No default provided')
parser.add_argument('-n3',    '--name3',     type=str,     default='old deepcore',             action='store', help='input file 3. No default provided')

args = parser.parse_args()
input1   = args.input1
input2  = args.input2
input3   = args.input3
name1 = args.name1
name2 = args.name2
name3 = args.name3



# fileList = ['in1','in2']
fileList = ['in1','in2','in3']
parPull = ['Dxy','Dz','Phi','Pt','Theta'] #pull
parPullvsEta = ['h_dxy','h_dz','h_phi','h_pt','h_theta'] #pull vs eta
parPullvsPt = ['dxy','dz','phi','pt','theta'] #pull vs pt 
parRes = ['dxy','dz','phi','pt','cotTheta'] #res

kindDict = {
    'pull' : (parPull,['pull']),
    'pullHighPt' : (parPullvsPt,['pull_highPt200-700']),
    # 'pullvsEta' : (parPullvsEta,['pulleta_Mean','pulleta_Sigma']), #OLD RELESE, PRE 11_2_X
    # 'pullvsPt' : (parPullvsPt,['pull_vs_pt_Mean','pull_vs_pt_Sigma']),  #OLD RELESE, PRE 11_2_X
    'pullDiff' : (parPullvsEta,['pulleta_Mean','pulleta_Sigma', 'pullpt_Mean','pullpt_Sigma']),
    'res' : (parRes,['res_vs_eta','res_vs_pt_Mean','res_vs_pt_Sigma', 'res_vs_eta_Mean','res_vs_eta_Sigma', 'res_vs_pt']),
    'chi2' : (['chi2'],['','_prob','mean','mean_vs_drj','mean_vs_pt']),
}

pathDict = {
    # 'generalTracks' : "DQMData/Run 1/Tracking/Run summary/Track/general_MTVAssociationByChi2/",
    # 'jetCoreTracks' :"DQMData/Run 1/Tracking/Run summary/Track/cutsRecoJetCoreRegionalStepByOriginalAlgo_MTVAssociationByChi2/"
    'generalTracks' : "DQMData/Run 1/Tracking/Run summary/JetCore/general_trackAssociatorByChi2/",
    'jetCoreTracks' :"DQMData/Run 1/Tracking/Run summary/JetCore/cutsRecoJetCoreRegionalStep_trackAssociatorByChi2/"
}



colorDict = {}
colorDict['in2'] = ROOT.kRed+2 #red
colorDict['in1'] = ROOT.kBlue+2 #blue
colorDict['in3'] = ROOT.kGreen+2 #green


fileDict = {}
fileDict['in1'] = ROOT.TFile.Open(input1)
fileDict['in2'] = ROOT.TFile.Open(input2)
fileDict['in3'] = ROOT.TFile.Open(input3)


legLabel = {
   'in1'  : name1,
   'in2'  : name2,
   'in3'  : name3,
}


def findmax(histo1,histo2) :
    val1 = histo1.GetMaximum()
    val2 = histo2.GetMaximum()
    if val1>val2 : 
        return val1
    else : 
        return val2
        
def findmin(histo1,histo2) :
    val1 = histo1.GetMinimum()
    val2 = histo2.GetMinimum()
    if val1<val2 : 
        return val1
    else : 
        return val2


histoDict = {}
for pind,path in pathDict.iteritems() :
    for f in fileList :
        for k,kval in kindDict.iteritems() :
            for par in kval[0] :
                for plot in kval[1] :     
                                   
                    if k=='pull' :
                        histoDict[pind+plot+par+f] = fileDict[f].Get(path+plot+par)             
                    if k=='pullDiff' or k=='res' or k=='chi2': #k=='pullvsEta'
                        histoDict[pind+plot+par+f] = fileDict[f].Get(path+par+plot)
                        if plot=='res_vs_eta' : 
                            histoDict[pind+plot+par+f] = histoDict[pind+plot+par+f].ProjectionY(pind+'_'+plot+'_'+par+'_'+f,0,-1) #histoDict used like temph2
                        if plot=='res_vs_pt' : 
                            histoDict[pind+plot+par+f] = histoDict[pind+plot+par+f].ProjectionY(pind+'_'+plot+'_'+par+'_'+f,33,40) #histoDict used like temph2
                            histoDict[pind+plot+par+f].SetTitle(pind+'_'+plot+'_'+par+'(200-700 GeV)')
                            histoDict[pind+plot+par+f].SetName(pind+'_'+plot+'_'+par+'200-700GeV')        
                    if k=='pullHighPt' :
                        plotInFile = plot.replace('_highPt200-700','_vs_pt')
                        temph2 = fileDict[f].Get(path+par+plotInFile)
                        temph2.SetTitle(pind+'_'+plot+'_'+par+'(200-700 GeV)')
                        temph2.SetName(pind+'_'+plot+'_'+par)
                        histoDict[pind+plot+par+f] = temph2.ProjectionY(pind+'_'+plot+par+'_'+f,35,40)
                 
                    # if k=='pullvsPt' : #OLD RELESE, PRE 11_2_X
                    #     plotInFile = plot.replace('_Mean','').replace('_Sigma','')
                    #     temph2 = fileDict[f].Get(path+par+plotInFile)
                    #     temph2.SetTitle(pind+'_'+plot+'_'+par)
                    #     temph2.SetName(pind+'_'+plot+'_'+par)
                    #     if 'Mean' in plot :
                    #         histoDict[pind+plot+par+f] = temph2.ProfileX(pind+'_'+plot+par+'_'+f,0,-1)
                    #     if 'Sigma' in plot :
                    #         histoDict[pind+plot+par+f] = temph2.ProjectionX(pind+'_'+plot+'_'+par+'_'+f,0,-1)
                    #         for xx in range(1, temph2.GetNbinsX()+1) :
                    #             # tempSlice = temph2.ProjectionX(pind+'_'+plot+'_'+par+'_'+f+'_slice'+str(xx),xx,xx)
                    #             tempSlice = temph2.ProjectionY(pind+'_'+plot+'_'+par+'_'+f+'_slice'+str(xx),xx,xx)
                    #             histoDict[pind+plot+par+f].SetBinContent(xx,tempSlice.GetStdDev())
                    #             histoDict[pind+plot+par+f].SetBinError(xx,tempSlice.GetStdDevError())
                    histoDict[pind+plot+par+f].SetName(pind+'_'+plot+'_'+par+'_'+f)
                    histoDict[pind+plot+par+f].SetTitle(pind+' '+par+' '+plot)
                    histoDict[pind+plot+par+f].SetLineWidth(3)
                    histoDict[pind+plot+par+f].SetLineColor(colorDict[f])
                    histoDict[pind+plot+par+f].SetStats(0)

output= ROOT.TFile("residual_comparison.root","recreate")

#canvas
canvasDict = {}
# cloneDict = {} #dizionario per plottare gli istogrammi 2 volte con y axis diversi perche root e una merda
for pind,path in pathDict.iteritems() :
    for k,kval in kindDict.iteritems() :
            for par in kval[0] :
                for plot in kval[1] :
                    if par=='eta' and '_vs_pt' in plot : continue
                    canvasDict['leg'+pind+plot+par] = ROOT.TLegend(0.7,0.8,0.95,0.95) 
                    canvasDict[pind+plot+par] = ROOT.TCanvas('c_CRcomp_'+pind+'_'+plot+'_'+par,'c_CRcomp_'+pind+'_'+plot+'_'+par,800,600)
                    canvasDict[pind+plot+par].cd()
                    canvasDict[pind+plot+par].SetGridx()
                    canvasDict[pind+plot+par].SetGridy()
                    sameFlag = ''
                    maxval = findmax(histo1=histoDict[pind+plot+par+fileList[0]],histo2=histoDict[pind+plot+par+fileList[1]])
                    minval = 0 
                    if 'Mean' in plot: 
                        minval = findmin(histo1=histoDict[pind+plot+par+fileList[0]],histo2=histoDict[pind+plot+par+fileList[1]])
                    for f in fileList :
                        histoDict[pind+plot+par+f].GetYaxis().SetRangeUser(minval-0.1*abs(minval),maxval+0.1*abs(maxval))
                        if 'vs_drj' in plot :
                            histoDict[pind+plot+par+f].GetXaxis().SetRangeUser(0.001,0.1)
                            # canvasDict[pind+plot+par].SetLogx()
                        histoDict[pind+plot+par+f].Draw(sameFlag)
                        sameFlag = 'SAME'
                        canvasDict['leg'+pind+plot+par].AddEntry(histoDict[pind+plot+par+f], legLabel[f])#+', #sigma=%.2f'%histoDict[f+par+t+k].GetStdDev())
                    canvasDict['leg'+pind+plot+par].Draw("SAME")

#writing
output.cd()
for pind,path in pathDict.iteritems() :
    for f in fileList :
        for k,kval in kindDict.iteritems() :
            for par in kval[0] :
                for plot in kval[1] :
                    # if par=='eta' and '_vs_pt' in plot : continue
                    histoDict[pind+plot+par+f].Write()
for pind,path in pathDict.iteritems() :
    for k,kval in kindDict.iteritems() :
            for par in kval[0] :
                for plot in kval[1] :
                    # if par=='eta' and '_vs_pt' in plot : continue
                    canvasDict[pind+plot+par].Write()
                    canvasDict[pind+plot+par].SaveAs("./"+pind+'_'+plot+'_'+par+".pdf")
                    canvasDict[pind+plot+par].SaveAs("./"+pind+'_'+plot+'_'+par+".png")


