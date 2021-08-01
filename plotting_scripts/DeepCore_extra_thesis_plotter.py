import math
import collections
import ROOT
from ROOT import gStyle
ROOT.gROOT.SetBatch(True)
gStyle.SetOptStat(0)
gStyle.SetPaintTextFormat(".3f");

cmslab = "#bf{CMS} #scale[0.7]{#it{Simulation Work in progress}}"
lumilab = " #scale[0.7]{13 TeV}"
cmsLatex = ROOT.TLatex()


############################## plot residuals ##############################################

inFile = ROOT.TFile.Open("seedValidation_JCcollection.root")

standard_seed_res_dxy = inFile.Get("standard_seed_res_dxy")
deepCore_seed_res_dxy = inFile.Get("deepCore_seed_res_dxy")

standard_seed_res_dxy.Rebin(20)
deepCore_seed_res_dxy.Rebin(20)

deepCore_seed_res_dxy.SetFillStyle(3003)
standard_seed_res_dxy.SetFillStyle(3004)
deepCore_seed_res_dxy.SetFillColor(ROOT.kRed-7)
standard_seed_res_dxy.SetFillColor(ROOT.kBlue-7)

# deepCore_seed_res_dxy.SetLineColor(ROOT.kRed-4)
# deepCore_seed_res_dxy.SetLineColor(ROOT.kBlue-4)

# deepCore_seed_res_dxy.GetXaxis().SetTitle("\Delta d_{xy} [cm]")
deepCore_seed_res_dxy.GetXaxis().SetTitle("Seed d_{xy}^{reco}-d_{xy}^{sim}  [cm]")
deepCore_seed_res_dxy.GetXaxis().SetTitleSize(0.045)
deepCore_seed_res_dxy.GetYaxis().SetTitle("Number of seeds")
# deepCore_seed_res_dxy.SetTitle("Seed d_{xy} residuals")
deepCore_seed_res_dxy.SetTitle('')
deepCore_seed_res_dxy.GetXaxis().SetRangeUser(-0.3,0.3)
standard_seed_res_dxy.GetXaxis().SetRangeUser(-0.3,0.3)
# deepCore_seed_res_dxy.GetYaxis().SetRangeUser(0,max(standard_seed_res_dxy.GetMaximum()*1.2,2000))
deepCore_seed_res_dxy.GetYaxis().SetRangeUser(0,800)


c_seed_res_dxy = ROOT.TCanvas("c_seed_res_dxy", "c_seed_res_dxy", 1600,1200)
c_seed_res_dxy.cd()
c_seed_res_dxy.SetGridy()
c_seed_res_dxy.SetGridx()
deepCore_seed_res_dxy.Draw()
standard_seed_res_dxy.Draw("same")
deepCore_seed_res_dxy.SetStats(0)
standard_seed_res_dxy.SetStats(0)

leg_seed_res_dxy = ROOT.TLegend(0.48, 0.73,0.88,0.88)
leg_seed_res_dxy.AddEntry(deepCore_seed_res_dxy,'DeepCore, #sigma='+str(round(deepCore_seed_res_dxy.GetStdDev(),3))+' cm')
leg_seed_res_dxy.AddEntry(standard_seed_res_dxy,'standard jetCore, #sigma='+str(round(standard_seed_res_dxy.GetStdDev(),3))+' cm')
leg_seed_res_dxy.Draw("same")

cmsLatex.SetNDC()
cmsLatex.SetTextFont(42)
cmsLatex.SetTextColor(ROOT.kBlack)
cmsLatex.SetTextAlign(31) 
cmsLatex.DrawLatex(1-c_seed_res_dxy.GetRightMargin(),1-0.8*c_seed_res_dxy.GetTopMargin(),lumilab)
cmsLatex.SetTextAlign(11) 
cmsLatex.DrawLatex(c_seed_res_dxy.GetLeftMargin(),1-0.8*c_seed_res_dxy.GetTopMargin(),cmslab)
cmsLatex.DrawLatex(c_seed_res_dxy.GetLeftMargin()+0.02,1-c_seed_res_dxy.GetTopMargin()-0.07,"QCD events (no PU)")
cmsLatex.DrawLatex(c_seed_res_dxy.GetLeftMargin()+0.02,1-c_seed_res_dxy.GetTopMargin()-0.14,"1.8 TeV <#hat p_{T}< 2.4 TeV")
cmsLatex.DrawLatex(c_seed_res_dxy.GetLeftMargin()+0.02,1-c_seed_res_dxy.GetTopMargin()-0.21,"p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4")

c_seed_res_dxy.SaveAs('c_seed_res_dxy.pdf')
c_seed_res_dxy.SaveAs('c_seed_res_dxy.png')











########################### timing plot ###########################

# timeDict = { #name : [standard, deepCore]
#     "jetCoreRegionalStep" :    	            [0.004494	, 0.002521          , ROOT.kGreen-7],
#     "jetCoreRegionalStepHitDoublets" :     	[0.000835	, 0         , ROOT.kMagenta-6],
#     "jetCoreRegionalStepSeedLayers" :      	[0.000068	, 0         , ROOT.kBlue-6],
#     "jetCoreRegionalStepSeeds" :       	    [0.003387	, 0.119963          , ROOT.kRed-7],
#     "jetCoreRegionalStepTrackCandidates" :  [0.900323	, 0.502637          , ROOT.kAzure+6],
#     "jetCoreRegionalStepTrackingRegions" :  [0.000021	, 0.000032          , ROOT.kYellow-3],
#     "jetCoreRegionalStepTracks" :      	    [0.0129	    , 0.007001          , ROOT.kSpring+8]
# }

timeDict = collections.OrderedDict()  #name : [standard, deepCore]
timeDict["jetCoreRegionalStep"] =    	            [0.004494	, 0.002521          , ROOT.kGreen-7]
timeDict["jetCoreRegionalStepHitDoublets"] =     	[0.000835	, 0         , ROOT.kMagenta-6]
timeDict["jetCoreRegionalStepSeedLayers"] =      	[0.000068	, 0         , ROOT.kBlue-6]
timeDict["jetCoreRegionalStepSeeds"] =       	    [0.003387	, 0.119963          , ROOT.kRed-7]
timeDict["jetCoreRegionalStepTrackCandidates"] =  [0.900323	, 0.502637          , ROOT.kAzure+6]
timeDict["jetCoreRegionalStepTrackingRegions"] =  [0.000021	, 0.000032          , ROOT.kYellow-3]
timeDict["jetCoreRegionalStepTracks"] =      	    [0.0129	    , 0.007001          , ROOT.kSpring+8]


href = ROOT.TH1F('href', 'href',8,0,8)

hdict = {}
for k,val in timeDict.items() :
    hdict[k] = href.Clone(k)
    hdict[k].SetBinContent(2,val[0])
    hdict[k].SetBinContent(5,val[1])
    hdict[k].SetFillColor(val[2])
    hdict[k].SetLineColor(1)
    for i in range(1, hdict[k].GetNbinsX()+1) :
        hdict[k].GetXaxis().SetBinLabel(i,'')
        if i==2 or i==5 : continue 
        hdict[k].SetBinContent(i,0)
    
    # hdict[k].GetXaxis().SetBinLabel(2,'standard jetCore')
    # hdict[k].GetXaxis().SetBinLabel(5,'DeepCore')
    # hdict[k].GetYaxis().SetTitle('Time [s/event]')
    # # hdict[k].GetXaxis().SetTickSize(0)
    # hdict[k].GetXaxis().SetTickLength(20)
    # hdict[k].GetXaxis().LabelsOption('h')
    # hdict[k].GetYaxis().SetRangeUser(0,1.5)
    # # hdict[k].SetLineWidth(0)
    # href.GetXaxis().SetLabelSize(0.07)
    # href.GetYaxis().SetRangeUser(0,1.1)



for k,val in timeDict.items() :
    href.Add(hdict[k])
for i in range(1, href.GetNbinsX()+1) :
    href.GetXaxis().SetBinLabel(i,'')
    if i==2 or i==5 : continue 
    href.SetBinContent(i,0)
    
href.GetXaxis().SetBinLabel(2,'standard jetCore')
href.GetXaxis().SetBinLabel(5,'DeepCore')
href.GetYaxis().SetTitle('Time [s/event]')
href.GetYaxis().SetTitleSize(0.05)
href.GetYaxis().SetTitleOffset(0.9)
href.GetYaxis().SetLabelSize(0.045)
href.GetYaxis().SetNdivisions(11)
href.GetXaxis().SetTickLength(0)
href.GetXaxis().LabelsOption('h')
# href.SetLineWidth(0) 
href.GetYaxis().SetRangeUser(0,1.1)
href.GetXaxis().SetLabelSize(0.07)
# href.SetTitle('Timing comparison')
href.SetTitle('')
href.SetMarkerSize(2)


timeStack = ROOT.THStack('timeStack','Timing comparison')
leg_time = ROOT.TLegend(0.35, 0.6,0.88,0.88)
for k,val in timeDict.items() :
    timeStack.Add(hdict[k])

for k,val in reversed(timeDict.items()) :
    
    leg_time.AddEntry(hdict[k],k.replace('jetCoreRegionalStep','jetCoreRegionalStep '))
# timeStack.SetLineWidth(2)
# timeStack.SetTtitle('Timing comparison')



c_time = ROOT.TCanvas("c_time", "c_time", 1600,1200)
c_time.SetGridy()
href.Draw('text0')

timeStack.Draw("same")
# timeStack.Draw("text0 same")
leg_time.Draw("same")

# hdict["jetCoreRegionalStepSeeds"].Draw("same text0")
# hdict["jetCoreRegionalStepSeeds"].SetMarkerSize(1.5)
# hdict["jetCoreRegionalStepSeeds"].SetBarOffset(-0.5)
# hdict["jetCoreRegionalStepTrackCandidates"].Draw("same text0")
# hdict["jetCoreRegionalStepSeeds"].SetMarkerSize(1.5)
# hdict["jetCoreRegionalStepSeeds"].SetBarOffset(-0.5)

cmsLatex.SetNDC()
cmsLatex.SetTextFont(42)
cmsLatex.SetTextColor(ROOT.kBlack)
cmsLatex.SetTextAlign(31) 
cmsLatex.DrawLatex(1-c_time.GetRightMargin(),1-0.8*c_time.GetTopMargin(),lumilab)
cmsLatex.SetTextAlign(11) 
cmsLatex.DrawLatex(c_time.GetLeftMargin(),1-0.8*c_time.GetTopMargin(),cmslab)
cmsLatex.SetTextAlign(31) 
cmsLatex.DrawLatex(1-c_time.GetRightMargin()-0.01,c_seed_res_dxy.GetBottomMargin()+0.39,"#scale[0.8]{QCD events (no PU)}")
cmsLatex.DrawLatex(1-c_time.GetRightMargin()-0.01,c_seed_res_dxy.GetBottomMargin()+0.32,"#scale[0.8]{1.8 TeV <#hat p_{T}< 2.4 TeV}")
cmsLatex.DrawLatex(1-c_time.GetRightMargin()-0.01,c_seed_res_dxy.GetBottomMargin()+0.25,"#scale[0.8]{p_{T}^{jet}>1 TeV, |#eta^{jet}|<1.4}")



c_time.SaveAs('c_time.pdf')
c_time.SaveAs('c_time.png')



outFile = ROOT.TFile('DeepCore_extra_thesis_plot.root', "recreate")
outFile.cd()
c_seed_res_dxy.Write()
c_time.Write()
outFile.Close()
