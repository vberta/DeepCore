from ROOT import *
from ROOT import gStyle
# from ROOT import TPad
import os
import argparse
import array as arr
gROOT.Reset()
gROOT.SetBatch(True)
gStyle.SetOptStat(0)

easyPlot = False #if true get already calulated plots, if false evaluate FR and Eff
perfectPlot = True #add perfect seed line
rebin_factor=5
rebindrj = False #merge first 3 bins in one
stack_sum = False #produce the hists as sum of iterations
byPosPerfNoJ = False
dedVal = { # dedicated validation directory (easyPlot not supported,byPosPerfNoJ )
    'NN'   : False,
    'Stand' : False,
    'NoJ'   : True,
    'Perf'  : True
    }
assocDedVal = 'trackAssociatorByChi2'    
# labelRatioDict = {
#     'eff' : "(Eff-Eff^{MC}) / Eff^{MC}",
#     'fake' : "(Fake-Fake^{MC}) / Fake^{MC}",
#     'dupl' : "(Dup-Dup^{MC}) / Dup^{MC}"
# } 
labelRatioDict = {
    'eff' : "(Eff-Eff^{DC113}) / Eff^{DC113}",
    'fake' : "(Fake-Fake^{DC113}) / Fake^{DC113}",
    'dupl' : "(Dup-Dup^{DC113}) / Dup^{DC113}"
} 
      



parser = argparse.ArgumentParser("plotter")
parser.add_argument('-inputNN',      '--inputNN',       type=str,                                           action='store', help='input file NN. No default provided')
parser.add_argument('-inputStand',   '--inputStand',    type=str,                                           action='store', help='input file standard. No default provided')
parser.add_argument('-inputPerf',    '--inputPerf',     type=str,                                           action='store', help='input file PerfectSeed. No default provided')
parser.add_argument('-inputNoJ',     '--inputNoJ',      type=str,                                           action='store', help='input file no jetcore. No default provided')
parser.add_argument('-assoc',        '--assoc',         type=str,   default='MTVAssociationByChi2',         action='store', help='associator name in file.')
parser.add_argument('-assocName',    '--assocName',     type=str,   default='chi2',                         action='store', help='associator name for the output.')

parser.add_argument('-nameNN',      '--nameNN',       type=str,     default='DeepCore',                     action='store', help='input file NN. No default provided')
parser.add_argument('-nameStand',   '--nameStand',    type=str,     default='Standard JetCore',             action='store', help='input file standard. No default provided')
parser.add_argument('-namePerf',    '--namePerf',     type=str,     default='MC truth seeding',             action='store', help='input file PerfectSeed. No default provided')
parser.add_argument('-nameNoJ',     '--nameNoJ',      type=str,     default='Without JetCore',              action='store', help='input file no jetcore. No default provided')

args = parser.parse_args()
inputNN   = args.inputNN
inputStand  = args.inputStand
inputPerf   = args.inputPerf
inputNoJ  = args.inputNoJ
assoc  = args.assoc
assocName  = args.assocName
nameNN = args.nameNN
nameStand = args.nameStand
namePerf = args.namePerf
nameNoJ = args.nameNoJ




myfile_NN = TFile(inputNN)
myfile_stand = TFile(inputStand)
myfile_perf = TFile(inputPerf)
myfile_noj = TFile(inputNoJ)


#GET HISTOS

if(easyPlot) :
    eff_stand = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/effic_vs_drj")
    fake_stand = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/fakerate_vs_drj")
    duplicate_stand = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/duplicatesRate_drj")

    eff_NN = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/effic_vs_drj")
    fake_NN = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/fakerate_vs_drj")
    duplicate_NN = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/duplicatesRate_drj")

    eff_perf = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/effic_vs_drj")
    fake_perf = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/fakerate_vs_drj")
    duplicate_perf = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/duplicatesRate_drj")

    eff_noj = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/effic_vs_drj")
    fake_noj = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/fakerate_vs_drj")
    duplicate_noj = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/duplicatesRate_drj")

else :

    edg_new = arr.array("d", [0.0025118869, 0.0039810711, 0.0063095726, 0.0099999998, 0.015848929, 0.025118863, 0.039810710, 0.063095726,0.1] )

    if not dedVal['Stand'] :
        eff_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(simToReco)_drj")
        eff_den_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_simul_drj")
        fake_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(recoToSim)_drj")
        fake_den_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
        duplicate_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_duplicate_drj")
    else :
        eff_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(simToReco)_drj")
        eff_den_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_simul_drj")
        fake_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(recoToSim)_drj")
        fake_den_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_reco_drj")
        duplicate_stand_TEMP = myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_duplicate_drj")
    # print("ent", duplicate_stand.GetEntries())
    duplicate_den_stand_TEMP= fake_den_stand_TEMP.Clone()   #myfile_stand.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")

    # edges_pre
    # Double_t edg_old[50] = { 0.0010000000, 0.0010964781, 0.0012022642, 0.0013182564, 0.0014454401, 0.0015848933, 0.0017378009, 0.0019054606, 0.0020892958, 0.0022908670, 0.0025118869, 0.0027542291, 0.0030199517, 0.0033113111, 0.0036307801, 0.0039810711, 0.0043651569, 0.0047863019, 0.0052480749, 0.0057543991, 0.0063095726, 0.0069183083, 0.0075857779, 0.0083176391, 0.0091201095, 0.0099999998, 0.010964781, 0.012022642, 0.013182567, 0.014454396, 0.015848929, 0.017378008, 0.019054607, 0.020892957, 0.022908676, 0.025118863, 0.027542284, 0.030199518, 0.033113111, 0.036307801, 0.039810710, 0.043651581, 0.047863003, 0.052480735, 0.057543993, 0.063095726, 0.069183081, 0.075857759, 0.083176367, 0.091201067 }
    # Double_t edg_after_rebin5[10] = { 0.0010000000, 0.0015848933, 0.0025118869, 0.0039810711, 0.0063095726, 0.0099999998, 0.015848929, 0.025118863, 0.039810710, 0.063095726 }
    eff_stand_TEMP.Sumw2()
    eff_den_stand_TEMP.Sumw2()
    fake_stand_TEMP.Sumw2()
    fake_den_stand_TEMP.Sumw2()
    duplicate_stand_TEMP.Sumw2()
    duplicate_den_stand_TEMP.Sumw2()
    eff_stand_TEMP.Rebin(rebin_factor)
    eff_den_stand_TEMP.Rebin(rebin_factor)
    fake_stand_TEMP.Rebin(rebin_factor)
    fake_den_stand_TEMP.Rebin(rebin_factor)
    duplicate_stand_TEMP.Rebin(rebin_factor)
    duplicate_den_stand_TEMP.Rebin(rebin_factor)

    if(rebindrj) :
        eff_stand_TEMP.Rebin(8,"eff_stand",edg_new)
        eff_den_stand_TEMP.Rebin(8,"eff_den_stand",edg_new)
        fake_stand_TEMP.Rebin(8,"fake_stand",edg_new)
        fake_den_stand_TEMP.Rebin(8,"fake_den_stand",edg_new)
        duplicate_stand_TEMP.Rebin(8,"duplicate_stand",edg_new)
        duplicate_den_stand_TEMP.Rebin(8,"duplicate_den_stand",edg_new)
    else :
        eff_stand= eff_stand_TEMP.Clone()
        eff_den_stand = eff_den_stand_TEMP.Clone()
        fake_stand = fake_stand_TEMP.Clone()
        fake_den_stand =fake_den_stand_TEMP.Clone()
        duplicate_stand = duplicate_stand_TEMP.Clone()
        duplicate_den_stand = duplicate_den_stand_TEMP.Clone()


    eff_stand.SetBinContent(1,eff_stand.GetBinContent(1)+eff_stand.GetBinContent(0))
    eff_den_stand.SetBinContent(1,eff_den_stand.GetBinContent(1)+eff_den_stand.GetBinContent(0))
    fake_stand.SetBinContent(1,fake_stand.GetBinContent(1)+fake_stand.GetBinContent(0))
    fake_den_stand.SetBinContent(1,fake_den_stand.GetBinContent(1)+fake_den_stand.GetBinContent(0))
    duplicate_stand.SetBinContent(1,duplicate_stand.GetBinContent(1)+duplicate_stand.GetBinContent(0))
    duplicate_den_stand.SetBinContent(1,duplicate_den_stand.GetBinContent(1)+duplicate_den_stand.GetBinContent(0))


    eff_stand.Divide(eff_den_stand)
    fake_stand.Add(fake_den_stand,fake_stand,1,-1)
    fake_stand.Divide(fake_den_stand)
    duplicate_stand.Divide(duplicate_den_stand)
    
    if not dedVal['NN'] :
        eff_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(simToReco)_drj")
        eff_den_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_simul_drj")
        fake_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(recoToSim)_drj")
        fake_den_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
        duplicate_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_duplicate_drj")
    else :
        eff_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(simToReco)_drj")
        eff_den_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_simul_drj")
        fake_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(recoToSim)_drj")
        fake_den_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_reco_drj")
        duplicate_NN_TEMP = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_duplicate_drj")
    duplicate_den_NN_TEMP = fake_den_NN_TEMP.Clone()#myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
    eff_NN_TEMP.Sumw2()
    eff_den_NN_TEMP.Sumw2()
    fake_NN_TEMP.Sumw2()
    fake_den_NN_TEMP.Sumw2()
    duplicate_NN_TEMP.Sumw2()
    duplicate_den_NN_TEMP.Sumw2()
    eff_NN_TEMP.Rebin(rebin_factor)#debug
    eff_den_NN_TEMP.Rebin(rebin_factor)#debug
    fake_NN_TEMP.Rebin(rebin_factor)
    fake_den_NN_TEMP.Rebin(rebin_factor)
    duplicate_NN_TEMP.Rebin(rebin_factor)
    duplicate_den_NN_TEMP.Rebin(rebin_factor)

    if(rebindrj) :
        eff_NN_TEMP.Rebin(8,"eff_NN",edg_new)
        eff_den_NN_TEMP.Rebin(8,"eff_den_NN",edg_new)
        fake_NN_TEMP.Rebin(8,"fake_NN",edg_new)
        fake_den_NN_TEMP.Rebin(8,"fake_den_NN",edg_new)
        duplicate_NN_TEMP.Rebin(8,"duplicate_NN",edg_new)
        duplicate_den_NN_TEMP.Rebin(8,"duplicate_den_NN",edg_new)
    else :
        eff_NN= eff_NN_TEMP.Clone()
        eff_den_NN = eff_den_NN_TEMP.Clone()
        fake_NN = fake_NN_TEMP.Clone()
        fake_den_NN =fake_den_NN_TEMP.Clone()
        duplicate_NN = duplicate_NN_TEMP.Clone()
        duplicate_den_NN = duplicate_den_NN_TEMP.Clone()

    eff_NN.SetBinContent(1,eff_NN.GetBinContent(1)+eff_NN.GetBinContent(0))
    eff_den_NN.SetBinContent(1,eff_den_NN.GetBinContent(1)+eff_den_NN.GetBinContent(0))
    fake_NN.SetBinContent(1,fake_NN.GetBinContent(1)+fake_NN.GetBinContent(0))
    fake_den_NN.SetBinContent(1,fake_den_NN.GetBinContent(1)+fake_den_NN.GetBinContent(0))
    duplicate_NN.SetBinContent(1,duplicate_NN.GetBinContent(1)+duplicate_NN.GetBinContent(0))
    duplicate_den_NN.SetBinContent(1,duplicate_den_NN.GetBinContent(1)+duplicate_den_NN.GetBinContent(0))

    eff_NN.Divide(eff_den_NN)#debug
    fake_NN.Add(fake_den_NN,fake_NN,1,-1)
    fake_NN.Divide(fake_den_NN)
    duplicate_NN.Divide(duplicate_den_NN)
    
    if byPosPerfNoJ :
        eff_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_assoc(simToReco)_drj")
        eff_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_simul_drj")
        fake_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_assoc(recoToSim)_drj")
        fake_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_reco_drj")
        duplicate_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_duplicate_drj")
    else :
        if not dedVal['Perf'] :
            eff_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(simToReco)_drj")
            eff_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_simul_drj")
            fake_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(recoToSim)_drj")
            fake_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
            duplicate_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_duplicate_drj")
        else :
            eff_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(simToReco)_drj")
            eff_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_simul_drj")
            fake_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(recoToSim)_drj")
            fake_den_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_reco_drj")
            duplicate_perf_TEMP = myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_duplicate_drj")
    duplicate_den_perf_TEMP = fake_den_perf_TEMP.Clone()#myfile_perf.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
    eff_perf_TEMP.Sumw2()
    eff_den_perf_TEMP.Sumw2()
    fake_perf_TEMP.Sumw2()
    fake_den_perf_TEMP.Sumw2()
    duplicate_perf_TEMP.Sumw2()
    duplicate_den_stand_TEMP.Sumw2()
    eff_perf_TEMP.Rebin(rebin_factor)
    eff_den_perf_TEMP.Rebin(rebin_factor)
    fake_perf_TEMP.Rebin(rebin_factor)
    fake_den_perf_TEMP.Rebin(rebin_factor)
    duplicate_perf_TEMP.Rebin(rebin_factor)
    duplicate_den_perf_TEMP.Rebin(rebin_factor)

    if(rebindrj) :
        eff_perf_TEMP.Rebin(8,"eff_perf",edg_new)
        eff_den_perf_TEMP.Rebin(8,"eff_den_perf",edg_new)
        fake_perf_TEMP.Rebin(8,"fake_perf",edg_new)
        fake_den_perf_TEMP.Rebin(8,"fake_den_perf",edg_new)
        duplicate_perf_TEMP.Rebin(8,"duplicate_perf",edg_new)
        duplicate_den_perf_TEMP.Rebin(8,"duplicate_den_perf",edg_new)
    else :
        eff_perf= eff_perf_TEMP.Clone()
        eff_den_perf = eff_den_perf_TEMP.Clone()
        fake_perf = fake_perf_TEMP.Clone()
        fake_den_perf =fake_den_perf_TEMP.Clone()
        duplicate_perf = duplicate_perf_TEMP.Clone()
        duplicate_den_perf = duplicate_den_perf_TEMP.Clone()

    eff_perf.SetBinContent(1,eff_perf.GetBinContent(1)+eff_perf.GetBinContent(0))
    eff_den_perf.SetBinContent(1,eff_den_perf.GetBinContent(1)+eff_den_perf.GetBinContent(0))
    fake_perf.SetBinContent(1,fake_perf.GetBinContent(1)+fake_perf.GetBinContent(0))
    fake_den_perf.SetBinContent(1,fake_den_perf.GetBinContent(1)+fake_den_perf.GetBinContent(0))
    duplicate_perf.SetBinContent(1,duplicate_perf.GetBinContent(1)+duplicate_perf.GetBinContent(0))
    duplicate_den_perf.SetBinContent(1,duplicate_den_stand.GetBinContent(1)+duplicate_den_stand.GetBinContent(0))

    eff_perf.Divide(eff_den_perf)
    fake_perf.Add(fake_den_perf,fake_perf,1,-1)
    fake_perf.Divide(fake_den_perf)
    duplicate_perf.Divide(duplicate_den_perf)

    if byPosPerfNoJ :
        eff_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_assoc(simToReco)_drj")
        eff_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_simul_drj")
        fake_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_assoc(recoToSim)_drj")
        fake_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_reco_drj")
        duplicate_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_duplicate_drj")
    else :
        if not dedVal['NoJ'] :
            eff_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(simToReco)_drj")
            eff_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_simul_drj")
            fake_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_assoc(recoToSim)_drj")
            fake_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
            duplicate_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_duplicate_drj")
        else :
            eff_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(simToReco)_drj")
            eff_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_simul_drj")
            fake_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_assoc(recoToSim)_drj")
            fake_den_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_reco_drj")
            duplicate_noj_TEMP = myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/JetCore/general_"+assocDedVal+"/num_duplicate_drj")
    duplicate_den_noj_TEMP = fake_den_noj_TEMP.Clone()#myfile_noj.Get("DQMData/Run 1/Tracking/Run summary/Track/general_"+assoc+"/num_reco_drj")
    eff_noj_TEMP.Sumw2()
    eff_den_noj_TEMP.Sumw2()
    fake_noj_TEMP.Sumw2()
    fake_den_noj_TEMP.Sumw2()
    duplicate_noj_TEMP.Sumw2()
    duplicate_den_noj_TEMP.Sumw2()
    eff_noj_TEMP.Rebin(rebin_factor)
    eff_den_noj_TEMP.Rebin(rebin_factor)
    fake_noj_TEMP.Rebin(rebin_factor)
    fake_den_noj_TEMP.Rebin(rebin_factor)
    duplicate_noj_TEMP.Rebin(rebin_factor)
    duplicate_den_noj_TEMP.Rebin(rebin_factor)

    if(rebindrj) :
        eff_noj_TEMP.Rebin(8,"eff_noj",edg_new)
        eff_den_noj_TEMP.Rebin(8,"eff_den_noj",edg_new)
        fake_noj_TEMP.Rebin(8,"fake_noj",edg_new)
        fake_den_noj_TEMP.Rebin(8,"fake_den_noj",edg_new)
        duplicate_noj_TEMP.Rebin(8,"duplicate_noj",edg_new)
        duplicate_den_noj_TEMP.Rebin(8,"duplicate_den_noj",edg_new)
    else :
        eff_noj= eff_noj_TEMP.Clone()
        eff_den_noj = eff_den_noj_TEMP.Clone()
        fake_noj = fake_noj_TEMP.Clone()
        fake_den_noj =fake_den_noj_TEMP.Clone()
        duplicate_noj = duplicate_noj_TEMP.Clone()
        duplicate_den_noj = duplicate_den_noj_TEMP.Clone()

    eff_noj.SetBinContent(1,eff_noj.GetBinContent(1)+eff_noj.GetBinContent(0))
    eff_den_noj.SetBinContent(1,eff_den_noj.GetBinContent(1)+eff_den_noj.GetBinContent(0))
    fake_noj.SetBinContent(1,fake_noj.GetBinContent(1)+fake_noj.GetBinContent(0))
    fake_den_noj.SetBinContent(1,fake_den_noj.GetBinContent(1)+fake_den_noj.GetBinContent(0))
    duplicate_noj.SetBinContent(1,duplicate_noj.GetBinContent(1)+duplicate_noj.GetBinContent(0))
    duplicate_den_noj.SetBinContent(1,duplicate_den_noj.GetBinContent(1)+duplicate_den_noj.GetBinContent(0))

    eff_noj.Divide(eff_den_noj)
    fake_noj.Add(fake_den_noj,fake_noj,1,-1)
    fake_noj.Divide(fake_den_noj)
    duplicate_noj.Divide(duplicate_den_noj)


    #debug -----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    #
    # def calculateEfficiency(tfile, num_name, denom_name, rebin=None,fake=False):
    #     num = tfile.Get(num_name)
    #     denom = tfile.Get(denom_name)
    #
    #     eff = num.Clone()
    #     den = denom.Clone()
    #     if(fake) :
    #         # den_tot_name = "DQMData/Run 1/Tracking/Run summary/Track/general_trackingParticleRecoAsssociation/num_reco_drj"
    #         den_tot_name = "DQMData/Run 1/Tracking/Run summary/Track/general_MTVAssociationByChi2/num_reco_drj"
    #         den_tot = tfile.Get(den_tot_name)
    #     else :
    #         den_tot = denom.Clone()
    #     if rebin is not None:
    #         eff.Rebin(rebin)
    #         den.Rebin(rebin)
    #         den_tot.Rebin(rebin)
    #     if(fake) :
    #         eff.Add(den,eff,1,-1)
    #     eff.Divide(den_tot)
    #     # eff.Divide(den)
    #     return eff
    #
    #
    # eff_NN = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/cutsRecoInitialStepByOriginalAlgoHp_MTVAssociationByChi2/num_assoc(simToReco)_drj")
    # eff_den_NN  = myfile_NN.Get("DQMData/Run 1/Tracking/Run summary/Track/cutsRecoInitialStepByOriginalAlgoHp_MTVAssociationByChi2/num_simul_drj")
    # eff_NN.Rebin(rebin_factor)
    # eff_den_NN.Rebin(rebin_factor)
    # eff_NN.Divide(eff_den_NN)
    #
    # folder_track = "DQMData/Run 1/Tracking/Run summary/Track/cutsReco%sByOriginalAlgo_MTVAssociationByChi2/"
    #
    # iterations = [
    #     "HighPtTripletStep",
    #     "LowPtQuadStep",
    #     "LowPtTripletStep",
    #     "DetachedQuadStep",
    #     "DetachedTripletStep",
    #     "MixedTripletStep",
    #     "PixelPairStep",
    #     "PixelLessStep",
    #     "TobTecStep",
    #     "JetCoreRegionalStep",
    #     "MuonSeededStepInOut",
    #     "MuonSeededStepOutIn",
    # ]
    # for it in iterations :
    #     num_name = folder_track%it + "num_assoc(simToReco)_drj"
    #     denom_name = folder_track%it + "num_simul_drj"
    #     eff = calculateEfficiency(myfile_NN, num_name, denom_name, rebin=rebin_factor)
    #     eff_NN.Add(eff)

    #END OF DEBUG ----------------------------------------------------------------------------------------------------










#HISTO PROPERTIES


eff_stand.SetLineColor(860+7)#azure
eff_NN.SetLineColor(800+10)#orange
eff_perf.SetLineColor(416+2) #green
eff_noj.SetLineColor(600-4)#blue

eff_stand.SetLineWidth(2)
eff_NN.SetLineWidth(3)
eff_perf.SetLineWidth(3)
eff_noj.SetLineWidth(3)

eff_stand.SetFillColor(860+6)#azure
# eff_NN.SetFillColorAlpha(800+7,0.00)#orange
# eff_perf.SetFillColorAlpha(416-7,0.00) #green
# eff_noj.SetFillColorAlpha(600-7,0.00)#blue

fake_stand.SetLineColor(860+7)#azure
fake_NN.SetLineColor(800+10)#orange
fake_perf.SetLineColor(416+2) #green
fake_noj.SetLineColor(600-4)#blue

fake_stand.SetFillColor(860+6)#azure
# fake_NN.SetFillColorAlpha(800+7,0.00)#orange
# fake_perf.SetFillColorAlpha(416-7,0.00) #green
# fake_noj.SetFillColorAlpha(600-7,0.00)#blue

fake_stand.SetLineWidth(2)
fake_NN.SetLineWidth(3)
fake_perf.SetLineWidth(3)
fake_noj.SetLineWidth(3)


duplicate_stand.SetLineColor(860+7)#azure
duplicate_NN.SetLineColor(800+10)#orange
duplicate_perf.SetLineColor(416+2) #green
duplicate_noj.SetLineColor(600-4)#blue

duplicate_stand.SetFillColor(860+6)#azure
# duplicate_NN.SetFillColorAlpha(800+7,0.00)#orange
# duplicate_perf.SetFillColorAlpha(416-7,0.00) #green
# duplicate_noj.SetFillColorAlpha(600-7,0.00)#blue

duplicate_stand.SetLineWidth(2)
duplicate_NN.SetLineWidth(3)
duplicate_perf.SetLineWidth(3)
duplicate_noj.SetLineWidth(3)

eff_stand.GetXaxis().SetTitle("#Delta R(track, jet)")
eff_stand.GetXaxis().SetTitleSize(0.04)
eff_stand.GetXaxis().SetTitleOffset(0.9)
eff_stand.GetYaxis().SetTitle("Tracking Efficiency")
eff_stand.SetTitle("Tracking Efficiency")
eff_stand.GetYaxis().SetRangeUser(0.6,1)

fake_stand.GetXaxis().SetTitle("#Delta R(track, jet)")
fake_stand.GetXaxis().SetTitleSize(0.04)
fake_stand.GetXaxis().SetTitleOffset(0.9)
fake_stand.GetYaxis().SetTitle("Fake Rate")
fake_stand.SetTitle("Fake Rate")
fake_stand.GetYaxis().SetRangeUser(0,0.22)

duplicate_stand.GetXaxis().SetTitle("#Delta R(track, jet)")
duplicate_stand.GetXaxis().SetTitleSize(0.04)
duplicate_stand.GetXaxis().SetTitleOffset(0.9)
duplicate_stand.GetYaxis().SetTitle("Duplicate Rate")
duplicate_stand.SetTitle("Duplicate Rate")
duplicate_stand.GetYaxis().SetRangeUser(0,0.11)


#PULL BUILDING

#eff
eff_stand_pull = eff_stand.Clone()
eff_stand_pull.Add(eff_perf,-1)
eff_stand_pull.Divide(eff_perf)
eff_stand_pull.SetFillStyle(0)
# eff_stand_pull.SetFillColor(860+6)#azure
eff_stand_pull.GetYaxis().SetRangeUser(-0.05,0.001)
eff_stand_pull.GetYaxis().SetTitle(labelRatioDict['eff'])
eff_stand_pull.GetYaxis().SetTitleSize(0.1)
eff_stand_pull.GetYaxis().SetLabelSize(0.1)
eff_stand_pull.GetYaxis().SetTitleOffset(0.5)
eff_stand_pull.SetTitle("")
eff_stand_pull.GetXaxis().SetTitleSize(0.1)
eff_stand_pull.GetXaxis().SetLabelSize(0.1)
eff_stand_pull.GetXaxis().SetTitleOffset(0.95)
# eff_stand.GetXaxis().SetLabelOffset(-0.005)
eff_stand.GetXaxis().SetLabelOffset(3)
eff_NN_pull = eff_NN.Clone()
eff_NN_pull.Add(eff_perf,-1)
eff_NN_pull.Divide(eff_perf)
eff_noj_pull = eff_noj.Clone()
eff_noj_pull.Add(eff_perf,-1)
eff_noj_pull.Divide(eff_perf)

#fake
fake_stand_pull = fake_stand.Clone()
fake_stand_pull.Add(fake_perf,-1)
fake_stand_pull.Divide(fake_perf)
fake_stand_pull.SetFillStyle(0)
# fake_stand_pull.SetFillColor(860+6)#azure
fake_stand_pull.GetYaxis().SetRangeUser(-0.04,0.6)
fake_stand_pull.GetYaxis().SetTitle(labelRatioDict['fake'])
fake_stand_pull.GetYaxis().SetTitleSize(0.08)
fake_stand_pull.GetYaxis().SetLabelSize(0.1)
fake_stand_pull.GetYaxis().SetTitleOffset(0.5)
fake_stand_pull.SetTitle("")
fake_stand_pull.GetXaxis().SetTitleSize(0.1)
fake_stand_pull.GetXaxis().SetLabelSize(0.1)
fake_stand_pull.GetXaxis().SetTitleOffset(0.95)
# fake_stand.GetXaxis().SetLabelOffset(-0.005)
fake_stand.GetXaxis().SetLabelOffset(3)
fake_NN_pull = fake_NN.Clone()
fake_NN_pull.Add(fake_perf,-1)
fake_NN_pull.Divide(fake_perf)
fake_noj_pull = fake_noj.Clone()
fake_noj_pull.Add(fake_perf,-1)
fake_noj_pull.Divide(fake_perf)

#duplicate
duplicate_stand_pull = duplicate_stand.Clone()
duplicate_stand_pull.Add(duplicate_perf,-1)
duplicate_stand_pull.Divide(duplicate_perf)
duplicate_stand_pull.SetFillStyle(0)
# duplicate_stand_pull.SetFillColor(860+6)#azure
duplicate_stand_pull.GetYaxis().SetRangeUser(-0.15,0.135)
duplicate_stand_pull.GetYaxis().SetTitle(labelRatioDict['dupl'])
duplicate_stand_pull.GetYaxis().SetTitleSize(0.1)
duplicate_stand_pull.GetYaxis().SetLabelSize(0.1)
duplicate_stand_pull.GetYaxis().SetTitleOffset(0.5)
duplicate_stand_pull.SetTitle("")
duplicate_stand_pull.GetXaxis().SetTitleSize(0.1)
duplicate_stand_pull.GetXaxis().SetLabelSize(0.1)
duplicate_stand_pull.GetXaxis().SetTitleOffset(0.95)
# duplicate_stand.GetXaxis().SetLabelOffset(-0.005)
duplicate_stand.GetXaxis().SetLabelOffset(3)
duplicate_NN_pull = duplicate_NN.Clone()
duplicate_NN_pull.Add(duplicate_perf,-1)
duplicate_NN_pull.Divide(duplicate_perf)
duplicate_noj_pull = duplicate_noj.Clone()
duplicate_noj_pull.Add(duplicate_perf,-1)
duplicate_noj_pull.Divide(duplicate_perf)





latexCMS = TLatex()
latexCMS.SetTextSize(0.05)

#LEGENDS

# leg_eff = TLegend(0.6,0.2,0.88,0.4);
leg_eff = TLegend(0.12,0.58,0.4,0.72);
leg_eff.AddEntry(eff_noj,nameNoJ)
leg_eff.AddEntry(eff_stand,nameStand)
leg_eff.AddEntry(eff_NN,nameNN)
if(perfectPlot) :
    leg_eff.AddEntry(eff_perf,namePerf)

# leg_fake = TLegend(0.6,0.2,0.88,0.4);
leg_fake = TLegend(0.12,0.58,0.4,0.72);
leg_fake.AddEntry(fake_noj,nameNoJ)
leg_fake.AddEntry(fake_stand,nameStand)
leg_fake.AddEntry(fake_NN,nameNN)
if(perfectPlot) :
    leg_fake.AddEntry(fake_perf,namePerf)

leg_duplicate = TLegend(0.6,0.6,0.88,0.75);
leg_duplicate.AddEntry(duplicate_noj,nameNoJ)
leg_duplicate.AddEntry(duplicate_stand,nameStand)
leg_duplicate.AddEntry(duplicate_NN,nameNN)
if(perfectPlot) :
    leg_duplicate.AddEntry(duplicate_perf,"MC truth seeding")


#CANVAS Draw


#efficiency ------
c_eff = TCanvas("c_eff","c_eff",800,800)
pad_eff_histo = TPad("pad_eff_histo","c_eff",0,0.245,1,1)
pad_eff_pull = TPad("pad_eff_pull","c_eff",0,0,1,0.265)

c_eff.cd()
pad_eff_pull.SetTopMargin(0.999);
pad_eff_pull.SetBottomMargin(0.2);
pad_eff_pull.Draw()
pad_eff_pull.cd()
pad_eff_pull.SetLogx()
pad_eff_pull.SetGrid()
eff_stand_pull.Draw("hist")
eff_stand_pull.SetStats(0)
eff_NN_pull.Draw("hist SAME")
eff_noj_pull.Draw("hist SAME")

c_eff.cd()
# c_eff.SetLogx()
# c_eff.SetGrid()
pad_eff_histo.SetBottomMargin(0.03);
pad_eff_histo.Draw()
pad_eff_histo.cd()
pad_eff_histo.SetLogx()
pad_eff_histo.SetGrid()
eff_stand.Draw("hist")
eff_stand.SetStats(0)
eff_noj.Draw("hist SAME")
if(perfectPlot) :
    eff_perf.Draw("hist SAME")
eff_NN.Draw("hist SAME")
leg_eff.Draw("SAME")
latexCMS.DrawLatex(0.0011,1.005,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.97
latexCMS.DrawLatex(0.055,1.005,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0012,0.97,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.0012,0.94,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")


#fake ------
c_fake = TCanvas("c_fake","c_fake",800,800)
pad_fake_histo = TPad("pad_fake_histo","c_fake",0,0.245,1,1)
pad_fake_pull = TPad("pad_fake_pull","c_fake",0,0,1,0.26)

c_fake.cd()
pad_fake_pull.SetTopMargin(0.999);
pad_fake_pull.SetBottomMargin(0.2);
pad_fake_pull.Draw()
pad_fake_pull.cd()
pad_fake_pull.SetLogx()
pad_fake_pull.SetGrid()
fake_stand_pull.Draw("hist")
fake_stand_pull.SetStats(0)
fake_NN_pull.Draw("hist SAME")
fake_noj_pull.Draw("hist SAME")

c_fake.cd()
# c_fake.SetLogx()
# c_fake.SetGrid()
pad_fake_histo.SetBottomMargin(0.03);
pad_fake_histo.Draw()
pad_fake_histo.cd()
pad_fake_histo.SetLogx()
pad_fake_histo.SetGrid()
fake_stand.Draw("hist")
fake_stand.SetStats(0)
fake_noj.Draw("hist SAME")
if(perfectPlot) :
    fake_perf.Draw("hist SAME")
fake_NN.Draw("hist SAME")
leg_fake.Draw("SAME")
latexCMS.DrawLatex(0.0011,0.222,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.18
latexCMS.DrawLatex(0.055,0.222,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0012,0.207,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.0012,0.19,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")


#duplicate
c_duplicate = TCanvas("c_duplicate","c_duplicate",800,800)
pad_duplicate_histo = TPad("pad_duplicate_histo","c_duplicate",0,0.245,1,1)
pad_duplicate_pull = TPad("pad_duplicate_pull","c_duplicate",0,0,1,0.265)

c_duplicate.cd()
pad_duplicate_pull.SetTopMargin(0.999);
pad_duplicate_pull.SetBottomMargin(0.2);
pad_duplicate_pull.Draw()
pad_duplicate_pull.cd()
pad_duplicate_pull.SetLogx()
pad_duplicate_pull.SetGrid()
duplicate_stand_pull.Draw("hist")
duplicate_stand_pull.SetStats(0)
duplicate_NN_pull.Draw("hist SAME")
duplicate_noj_pull.Draw("hist SAME")

c_duplicate.cd()
# c_duplicate.SetLogx()
# c_duplicate.SetGrid()
pad_duplicate_histo.SetBottomMargin(0.03);
pad_duplicate_histo.Draw()
pad_duplicate_histo.cd()
pad_duplicate_histo.SetLogx()
pad_duplicate_histo.SetGrid()
duplicate_stand.Draw("hist")
duplicate_stand.SetStats(0)
duplicate_noj.Draw("hist SAME")
if(perfectPlot) :
    duplicate_perf.Draw("hist SAME")
duplicate_NN.Draw("hist SAME")
leg_duplicate.Draw("SAME")
latexCMS.DrawLatex(0.0011,0.111,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.015 0.89
latexCMS.DrawLatex(0.055,0.111,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0028,0.103,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.015,0.094,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")





c_eff.SaveAs("eff_"+assocName+"_comparison.png")
c_fake.SaveAs("fake_"+assocName+"_comparison.png")
c_duplicate.SaveAs("duplicate_"+assocName+"_comparison.png")
c_eff.SaveAs("eff_"+assocName+"_comparison.pdf")
c_fake.SaveAs("fake_"+assocName+"_comparison.pdf")
c_duplicate.SaveAs("duplicate_"+assocName+"_comparison.pdf")




#histogram for statistic error

c_stat_eff = TCanvas("c_stat_eff","c_stat_eff",800,600)
c_stat_eff.cd()
c_stat_eff.SetLogx()
c_stat_eff.SetGrid()
eff_stand.Draw("")
eff_noj.Draw(" SAME")
if(perfectPlot) :
    eff_perf.Draw(" SAME")
eff_NN.Draw(" SAME")
leg_eff.Draw("SAME")
latexCMS.DrawLatex(0.0011,1.005,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.97
latexCMS.DrawLatex(0.055,1.005,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0012,0.97,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.0012,0.94,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")



c_stat_fake = TCanvas("c_stat_fake","c_stat_fake",800,600)
c_stat_fake.cd()
c_stat_fake.SetLogx()
c_stat_fake.SetGrid()
fake_stand.Draw("")
fake_noj.Draw(" SAME")
if(perfectPlot) :
    fake_perf.Draw(" SAME")
fake_NN.Draw(" SAME")
leg_fake.Draw("SAME")
latexCMS.DrawLatex(0.0011,0.222,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.18
latexCMS.DrawLatex(0.055,0.222,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0012,0.207,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.0012,0.19,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")

c_stat_duplicate = TCanvas("c_stat_duplicate","c_stat_duplicate",800,600)
c_stat_duplicate.cd()
c_stat_duplicate.SetLogx()
c_stat_duplicate.SetGrid()
duplicate_stand.Draw("")
duplicate_noj.Draw(" SAME")
if(perfectPlot) :
    duplicate_perf.Draw(" SAME")
duplicate_NN.Draw(" SAME")
leg_duplicate.Draw("SAME")
latexCMS.DrawLatex(0.0011,0.111,"#bf{#bf{CMS}} #scale[0.7]{#bf{#it{Simulation Preliminary}}}") #0.015 0.89
latexCMS.DrawLatex(0.055,0.111,"#bf{13 TeV}")
latexCMS.DrawLatex(0.0028,0.103,"QCD 1800 GeV <#hat p_{T}< 2400 GeV (no PU)")
latexCMS.DrawLatex(0.015,0.094,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")

c_stat_eff.SaveAs("eff_"+assocName+"_comparison_STAT.png")
c_stat_fake.SaveAs("fake_"+assocName+"_comparison_STAT.png")
c_stat_duplicate.SaveAs("duplicate_"+assocName+"_comparison_STAT.png")
c_stat_eff.SaveAs("eff_"+assocName+"_comparison_STAT.pdf")
c_stat_fake.SaveAs("fake_"+assocName+"_comparison_STAT.pdf")
c_stat_duplicate.SaveAs("duplicate_"+assocName+"_comparison_STAT.pdf")



output = TFile("plot_"+assocName+".root","recreate")
output.cd()
c_eff.Write()
c_fake.Write()
c_duplicate.Write()
c_stat_eff.Write()
c_stat_fake.Write()
c_stat_duplicate.Write()
