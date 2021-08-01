from ROOT import *

# file_standard = TFile.Open("dqm_standard_chi2.root")
# file_deepcore = TFile.Open("dqm_deepcore_chi2.root")
# file_perfect = TFile.Open("dqm_perfect_chi2.root")
rebuild = False
rebComp = True #rebuild comparison activation
if rebComp : 
    rebuild = True
rebString = ''
if not rebuild :
    rebString = '_noRebuild'
if rebComp :
    rebString = '_rebComp'

file_standard = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/standard/btag_diffPt/DQM_standard.root")
if rebuild :
    file_deepcore = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/btag_diffPt/DQM_deepCore.root")
else :
    file_deepcore = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/btag_diffPt_noRebuild/DQM_deepcore_noRebuild.root")
if rebComp : file_deepcore_noreb = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/btag_diffPt_noRebuild/DQM_deepcore_noRebuild.root")
    # file_deepcore = TFile.Open("./../debug/DQM_noRebuild.root")
    # file_deepcore = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/noRebuild/DQM_deepcore_noRebuild.root")

file_perfect = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/perfect/btag_diffPt/DQM_perfect.root") #comment 
# file_perfect = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/dqm_deepcore_chi2.root") #uncomment
# file_perfect = TFile.Open("/scratchssd/bertacch/tracking/step34_20k/deepcore/btag_diffPt/DQM_deepCore.root") #uncomment
# file_perfect = TFile.Open("./../debug/DQM.root") #uncomment


# pt_range='GLOBAL'
# pt_range='PT_1500-3000'
# pt_range='PT_800-1500'
# pt_range='PT_500-800'

pt_rangeList = []
pt_rangeList.append('GLOBAL')
pt_rangeList.append('PT_1500-3000')
pt_rangeList.append('PT_800-1500')
pt_rangeList.append('PT_500-800')

# output =TFile("my_comparison_"+pt_range+".root", "recreate")
output =TFile("btag_comparison"+rebString+".root", "recreate")


for pt_range in pt_rangeList : 
# h_standard = file_standard.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_GLOBAL/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_GLOBAL")
# h_deepcore = file_deepcore.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_GLOBAL/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_GLOBAL")
# h_perfect = file_perfect.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_GLOBAL/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_GLOBAL")
    h_standard = file_standard.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_"+pt_range+"/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_"+pt_range)
    h_deepcore = file_deepcore.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_"+pt_range+"/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_"+pt_range)
    h_perfect = file_perfect.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_"+pt_range+"/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_"+pt_range)
    if rebComp : h_deepcore_noreb = file_deepcore_noreb.Get("DQMData/Run 1/Btag/Run summary/deepCSV_BvsAll_"+pt_range+"/FlavEffVsBEff_DUSG_discr_deepCSV_BvsAll_"+pt_range)



    h_deepcore.SetMarkerColor(2)
    h_perfect.SetMarkerColor(3)
    if rebComp : h_deepcore_noreb.SetMarkerColor(4)
    h_deepcore.SetLineColor(2)
    h_perfect.SetLineColor(3)
    if rebComp : h_deepcore_noreb.SetLineColor(4)
    
    # h_standard.SetLineWidth(1)
    # h_deepcore.SetLineWidth(1)
    # h_perfect.SetLineWidth(1)
    # if rebComp : h_deepcore_noreb.SetLineWidth(1)

    c_compare = TCanvas("c_compare"+pt_range,"c_compare"+pt_range)
    c_compare.cd()
    c_compare.SetGridx()
    c_compare.SetGridy()
    h_standard.Draw()
    h_deepcore.Draw("SAME")
    h_perfect.Draw("SAME")
    if rebComp : h_deepcore_noreb.Draw("SAME")

    leg_compare = TLegend(0.1,0.7,0.48,0.9)
    leg_compare.AddEntry(h_standard, "standard")
    leg_compare.AddEntry(h_deepcore, "deepcore")
    leg_compare.AddEntry(h_perfect, "perfect")
    if rebComp : leg_compare.AddEntry(h_deepcore_noreb, "deepcore,noRebuild")
    leg_compare.Draw("SAME")

    # output =TFile("my_comparison_"+pt_range+"_noRebuild.root", "recreate")
    c_compare.Write()
    h_standard.Write()
    h_deepcore.Write()
    h_perfect.Write()
    if rebComp : h_deepcore_noreb.Write()
    
raw_input('Press Enter to exit')
