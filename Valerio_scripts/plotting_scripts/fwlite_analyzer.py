#! /usr/bin/env python
from __future__ import division
from itertools  import combinations
import ROOT
import sys
from DataFormats.FWLite import Events, Handle
from math import *

# kind = 'deepcore/noRebuild'  
# eventTree = Events (['/scratchssd/bertacch/tracking/step34_20k/'+kind+'/step3.root'])

CHI2_CUT=25
invalidChi2 = 10000000000.

# fileName = 'withTP_JCcollection'
# fileName = 'noRebuild_withTP_JCcollection'
# fileName= 'standard'
fileName = 'backFit'
if fileName =='standard' :
    eventTree = Events (['/home/users/bertacch/cms/deepCore_studies/test_reintegrationStandard/CMSSW_10_5_0_pre2/src/test_seedValidation/test_1kev_withTP_JCcollections/step3.root'])
else :
    eventTree = Events (['/gpfs/ddn/users/bertacch/cms/CMSSW_10_5_0_pre2/src/test_deepCore_seedValidation/test_1kev_'+fileName+'/step3.root'])
# eventTree = Events (['/gpfs/ddn/users/bertacch/cms/CMSSW_10_5_0_pre2/src/test_deepCore_seedValidation/debug/step3.root'])
handleTRK = Handle ("std::vector<reco::Track>")
handleTRK_jetcore = Handle ("std::vector<reco::Track>")
handleSEED_jetcore = Handle ("std::vector<reco::Track>") #TrajectorySeed is the type of the seed, but these are the track-from-seed
handleTP = Handle ("std::vector<TrackingParticle>")   # TrackingParticleCollection
# handleTP = Handle ("edm::RefVector<TrackingParticleCollection>")   # TrackingParticleCollection
handleBS = Handle("reco::BeamSpot")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
labelTRK = ("generalTracks","","RECO")
labelTRK_jetcore = ("jetCoreRegionalStepTracks","","RECO")
# if fileName == 'standard' :
labelSEED_jetcore = ("seedTracksjetCoreRegionalStepSeeds","", "RECO")
# else :
#     labelSEED_jetcore = ("jetCoreSeeds","", "RECO")
labelTP = ("mix","MergedTrackTruth","HLT")
labelBS = ("offlineBeamSpot","","RECO")



#building histos
ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.gROOT.SetStyle('Plain') # white background
hist={}
parList = ['dxy','dz','phi','eta','ptinv']
levelList = ['seed','track']
kindList = ['res','pull']
varDict = {}
for level in levelList :
    varDict['nHit'+level] = ('nHit'+'_'+level,50,0,50)
    for par in parList :
        for kind in kindList :
            nbin = 100
            xlimit = 10
            if kind=='pull' :
                xlimit=xlimit/1
            if kind == 'res' :
                if par == 'dxy' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-0.3,0.3)
                if par == 'dz' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-1.,1.)
                if par == 'phi' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-0.1,0.1)
                if par == 'eta' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-0.05,0.05)
                if par == 'ptinv' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,4000,-0.5,0.5)
            if kind == 'pull' :
                if par == 'dxy' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-2,2)
                if par == 'dz' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-2,2)
                if par == 'phi' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-2,2)
                if par == 'eta' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-4,4)
                if par == 'ptinv' : varDict[par+level+kind] = (par+'_'+level+'_'+kind,1000,-0.4,0.4)
for i,val in varDict.iteritems() :
    hist[i] = ROOT.TH1F(val[0],val[0],val[1],val[2],val[3])

#chi2 associator copied from:
#trackValidation_cff -> trackingParticleRecoTrackAsssociation_cfi -> TrackAssociatorEDProducer -> trackAssociatorByChi2 ->
#-> trackAssociatorByChi2Producer -> TrackAssociatorByChi2Impl -> trackAssociationChi2 
#(in SimTracker/trackAssociation)
def getChi2(reco,sim) : #copied from 
    chi2 = invalidChi2
    if sim.charge()!=0 :
        recoPars = reco.parameters()
        recoCov = reco.covariance()
        recoCov.Invert()
        #here the approx printed!
        # print "vert=", sim.vertex().X(), sim.vx(), "phi,eta", sim.theta(), sim.momentum().Theta(),sim.phi(), sim.momentum().Phi()
        simPars = reco.parameters() #a copy to have the type ROOT.ROOT::Math::SVector<double,5>
        simPars[0] = sim.charge()/sim.p()
        simPars[1] = 0.5 * 3.141592653589793238 - sim.theta()
        simPars[2] = sim.phi()
        simPars[3] = -sim.vertex().x()
        simPars[3] = -sim.vx() * sin(sim.phi()) + sim.vy() * cos(sim.phi())
        simPars[4] = sim.vz() * sim.pt() / sim.p() - (sim.vx() * sim.px() + sim.vy() * sim.py()) / sim.pt() * sim.pz() / sim.p()
    
        diffPars = recoPars-simPars
        chi2 = ROOT.Math.Dot(diffPars*recoCov,diffPars)
        chi2 = chi2/5
    return chi2
    
# loop over events
print "WARNING: association ByChi2 done with an approximation: simTrack patameters evaluated at vertex, reco track at PCAbeamspot "
print fileName
count= 0
for event in eventTree:
    count+=1
    print count 
    # if count % 100 == 0 :
	#     print count
    # if count > 1 :
    #     break
    event.getByLabel (labelTRK, handleTRK)
    event.getByLabel (labelTRK_jetcore, handleTRK_jetcore)
    event.getByLabel (labelSEED_jetcore, handleSEED_jetcore)
    event.getByLabel (labelTP, handleTP)
    event.getByLabel (labelBS,handleBS)
    
    # get the product
    TRK = handleTRK.product()
    TRK_jetcore = handleTRK_jetcore.product()
    SEED_jetcore = handleSEED_jetcore.product()
    TP = handleTP.product()
    BS = handleBS.product()
    
    # print "trackPart=", len(TP), ", seed=", len(SEED_jetcore), ", track=", len(TRK_jetcore)
    
    recoObjDict ={
        'seed' : SEED_jetcore,
        'track' : TRK_jetcore,
    } 
    
    def sortChi2(sim) :
        return sim[1]
   
    # #debugn
    # nRecoAssoc = 0
    # nRecoFake= 0 
    # nRecoTot =0 
    # nSimTot =0
    # KDEB='seed'
    
    for o, recoObj in recoObjDict.iteritems() :
        for reco in recoObj :
            if reco.eta()<1.4 and reco.eta()>-1.4: #barrel only! 
                # if o==KDEB : nRecoTot +=1 #debugn
                # if o==KDEB : assocFlag = False #debugn
                Nhits = reco.hitPattern().numberOfValidPixelBarrelHits()
                hist['nHit'+o].Fill(Nhits)
                assocSim = []
                for sim in TP :
                    # if o==KDEB and nRecoTot==1 : nSimTot +=1#debugn
                    chi2 = getChi2(reco,sim)
                    if chi2<CHI2_CUT :
                        assocSim.append((sim,chi2))
                        # if o==KDEB : assocFlag = True #debugn
                if len(assocSim)>0 :

                    # if o==KDEB : nRecoAssoc +=1 #debugn
                    
                    assocSim.sort(key=sortChi2) 
                    sim = assocSim[0][0]
                        
                    sim_dxy = (-(sim.vx() - BS.x0()) * sim.py() + (sim.vy() - BS.y0()) * sim.px()) / sim.pt()
                    sim_dz = (sim.vz() - BS.z0()) -((sim.vx() - BS.x0()) * sim.px() + (sim.vy() - BS.y0()) * sim.py()) / sim.pt() * sim.pz() / sim.pt()
                    
                    reco_dxy = (-(reco.vx() - BS.x0()) * reco.py() + (reco.vy() - BS.y0()) * reco.px()) / reco.pt()
                    reco_dz = (reco.vz() - BS.z0()) -((reco.vx() - BS.x0()) * reco.px() + (reco.vy() - BS.y0()) * reco.py()) / reco.pt() * reco.pz() / reco.pt()
                    
                    
                    # print "debug associated", "nAssoc=",len(assocSim),  ", pdgId=",sim.pdgId(), ", other info=", sim.numberOfHits(), sim.pt(), sim.p(), sim.eta(), sim_dz
                    # print "---start new track--- n assoc=",len(assocSim), "kind=", o
                    # print "reco info: " , "ptRec=" , reco.pt() , ", dxyRec=" , reco_dxy , ", dzRec=" , reco_dz ,", etaRec=" , reco.eta(), "phiRec", reco.phi()
                    # print "sim info: " , "ptSim=" , sim.pt() , ", dxySim=" , sim_dxy , ", dzSim=" , sim_dz ,", etaSim=" , sim.eta() , "phisim", sim.phi()
                    # print "res info: " , "ptres=" , reco.pt()-sim.pt() , ", dxyRes=" , reco_dxy-sim_dxy , ", dzRes=" , reco_dz-sim_dz ,", etaRes=" , reco.eta()-sim.eta(), "phires", reco.phi()-sim.phi()
                
                    valDict = {}
                    # valDict['dxyres'] = reco.dxy()-sim_dxy
                    # valDict['dzres'] = reco.dz()-sim_dz
                    valDict['dxyres'] = reco_dxy-sim_dxy
                    valDict['dzres'] = reco_dz-sim_dz
                    valDict['phires'] = reco.phi()-sim.phi()
                    valDict['etares'] = reco.eta()-sim.eta()
                    if reco.pt()!=0 and sim.pt()!=0 :
                        valDict['ptinvres'] = reco.charge()/reco.pt()-sim.charge()/sim.pt()
                    else :
                        valDict['ptinvres'] =-9999 
                        
                    if reco.dxyError()!=0 : valDict['dxypull'] =valDict['dxyres']/reco.dxyError()
                    else : valDict['dxypull']=-9999
                    if reco.dzError()!=0 : valDict['dzpull'] = valDict['dzres']/reco.dzError()
                    else : valDict['dzpull']=-9999
                    if reco.phiError()!=0 : valDict['phipull'] =valDict['phires']/reco.phiError()
                    else : valDict['phipull']=-9999
                    if reco.etaError()!=0 : valDict['etapull'] =valDict['etares']/reco.etaError()
                    else : valDict['etapull']=-9999
                    if reco.pt()!=0 and reco.ptError()!=0 : 
                        err = reco.ptError()/reco.pt()
                        valDict['ptinvpull'] = valDict['ptinvres']/err
                    else : valDict['ptinvpull']=-9999
                    
                    for par in parList :
                        for kind in kindList : 
                            hist[par+o+kind].Fill(valDict[par+kind])
                # else :
                #     print "---start NOT RECO track--- n assoc=",len(assocSim) , "kind=", o
                #     print "reco info: " , "ptRec=" , reco.pt() , ", dxyRec=" , reco.dxy() , ", dzRec=" , reco.dz() ,", etaRec=" , reco.eta()
                    # if o==KDEB : nRecoFake +=1 #debugn
                    
     #debugn
    # print "n reco in acceptance=", nRecoTot,",nsim=", nSimTot
    # if nRecoFake!=0 :
    #     print "n reco associated=", nRecoAssoc, ", nnotassociated=", nRecoFake, ", eff=", nRecoAssoc/(nSimTot), ", fake=", nRecoFake/nRecoTot, ", pass/fail=", nRecoAssoc/nRecoFake   
    # else :
    #     print "n reco associated=", nRecoAssoc, ", nnotassociated=", nRecoFake, ", eff=", nRecoAssoc/(nSimTot), ", fake=", nRecoFake/nRecoTot           
                                     
      
    
 
    # #debugn
    # npass = 0
    # nfail= 0 
    # nrecoTot =0 
    # nrecoOut = 0  
    # nsimTot =0
    # for o, recoObj in recoObjDict.iteritems() :
    #     for reco in recoObj :
    #         if reco.eta()<1.4 and reco.eta()>-1.4: #barrel only! 
    #             if o=='track' : nrecoTot +=1 #debugn
    #             if o=='track' : assoc = False #debugn
    #             Nhits = reco.hitPattern().numberOfValidPixelBarrelHits()
    #             hist['nHit'+o].Fill(Nhits)
    #             for sim in TP :
    #                 if o=='track' and nrecoTot==1 : nsimTot +=1#debugn
    #                 chi2 = getChi2(reco,sim)
    #                 if chi2<CHI2_CUT :
    #                     if o=='track' : npass +=1 #debugn
    #                     assoc = True
                        
    #                     sim_dxy = (-(sim.vx() - BS.x0()) * sim.py() + (sim.vy() - BS.y0()) * sim.px()) / sim.pt()
    #                     sim_dz = (sim.vz() - BS.z0()) -((sim.vx() - BS.x0()) * sim.px() + (sim.vy() - BS.y0()) * sim.py()) / sim.pt() * sim.pz() / sim.pt()
                    
    #                     valDict = {}
    #                     valDict['dxyres'] = reco.dxy()-sim_dxy
    #                     valDict['dzres'] = reco.dz()-sim_dz
    #                     valDict['phires'] = reco.phi()-sim.phi()
    #                     valDict['etares'] = reco.eta()-sim.eta()
    #                     if reco.pt()!=0 and sim.pt()!=0 :
    #                         valDict['ptinvres'] = reco.charge()/reco.pt()-sim.charge()/sim.pt()
    #                     else :
    #                         valDict['ptinvres'] =-9999 
                            
    #                     # if o!='seed' :    #the if is a DEBUGGGG!!! (the content of the if no!)
    #                     if reco.dxyError()!=0 : valDict['dxypull'] =valDict['dxyres']/reco.dxyError()
    #                     else : valDict['dxypull']=-9999
    #                     if reco.dzError()!=0 : valDict['dzpull'] = valDict['dzres']/reco.dzError()
    #                     else : valDict['dzpull']=-9999
    #                     if reco.phiError()!=0 : valDict['phipull'] =valDict['phires']/reco.phiError()
    #                     else : valDict['phipull']=-9999
    #                     if reco.etaError()!=0 : valDict['etapull'] =valDict['etares']/reco.etaError()
    #                     else : valDict['etapull']=-9999
    #                     if reco.pt()!=0 and reco.ptError()!=0 : 
    #                         err = reco.ptError()/reco.pt()
    #                         valDict['ptinvpull'] = valDict['ptinvres']/err
    #                     else : valDict['ptinvpull']=-9999
                        
    #                     for par in parList :
    #                         for kind in kindList : 
    #                             # if o=='track' and kind=='pull': continue #DEBUGGGG!!!
    #                             hist[par+o+kind].Fill(valDict[par+kind])
    #             if o=='track' :
    #                 if not assoc : nfail +=1 #debugn
    #         else : 
    #             if o=='track' : nrecoOut +=1  #debugn
    #  #debugn
    # print "n reco in acceptance=", nrecoTot, ", n out of accepance=", nrecoOut, ", eff=", nrecoTot/(nrecoTot+nrecoOut), "nSim=", nsimTot
    # if nfail!=0 :
    #     print "n reco associated=", npass, ", nnotassociated=", nfail, ", eff=", npass/(nsimTot), ", fake=", nfail/nrecoTot, ", pass/fail=", npass/nfail   
    # else :
    #     print "n reco associated=", npass, ", nnotassociated=", nfail, ", eff=", npass/(nsimTot), ", fake=", nfail/nrecoTot           
                    


                                
output = ROOT.TFile("FWlite_histos_"+fileName+"_CMSSWaligned_3.root","recreate")
for o in levelList :
    hist['nHit'+o].Write()
for par in parList :
    for level in levelList :
        for kind in kindList :
            hist[par+level+kind].Write()
output.Close()
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    # for j1 in jets1  :
    #         if abs(j1.eta()) < 2.5 and j1.pt() > 1500 :
	# 	summed=0;
    #             for disc in discriminators :
    # 		   btag=j1.bDiscriminator(disc)
	# 	   summed+=btag
	# 	btag=summed
    #             for sample in samples :
    #                     if samples[sample](j1) :     
    #                         hist["%s_%s"%("sum",sample)].Fill(btag)
# inthist={}
# graphs={}
# for disc in ["sum"]:
#   for sample in samples :
#      name="%s_%s"%(disc,sample)
#      print name
#      inthist[name] = ROOT.TH1F (name,name, 1000, 0, 1)
#      cumulative=0.
#      for i in xrange(hist[name].GetNbinsX(),-1,-1) :
#           cumulative+=hist[name].GetBinContent(i)
#           inthist[name].SetBinContent(i,cumulative/hist[name].GetEntries())
#   for s1,s2 in combinations(samples.keys(),2):
#      name="%s_%s_vs_%s"%(disc,s1,s2)
#      print name
#      graphs[name] = ROOT.TGraph(1001)
#      for i in xrange(0,1001) : 
#        graphs[name].SetPoint(i,inthist["%s_%s"%(disc,s1)].GetBinContent(i),inthist["%s_%s"%(disc,s2)].GetBinContent(i))
# 
# 
# c1 = ROOT.TCanvas()
# i=0
# color=[ROOT.kRed,ROOT.kBlue]
# for disc in ["sum"] :
#   graphs["%s_b_vs_udsg"%disc].SetMarkerColor(color[i])
#   graphs["%s_b_vs_udsg"%disc].SetLineColor(color[i])
#   graphs["%s_b_vs_udsg"%disc].Draw("ALP" if i ==0 else "LPsame")
#   i+=1
# 
# c1.SaveAs("btagC1.root")
# c1.SaveAs("btag.png")
