# DeepCore

This repository contains the code related to DeepCore (the CNN-based approach for the seeding in high energy jet tracking, in CMS reconstruction), outside of CMSSW. Is mostrly related to the training step, and dedicated validation plotter.

More information about DeepCore can be found at: https://twiki.cern.ch/twiki/bin/view/CMSPublic/NNJetCoreAtCtD2019

This repository contains the following directories:

## training  : 

## keras_to_TF :

## plotting_scripts :

## old development : 



<!--- A raw description of the directory tree (or particular branches):


## trackjet directory:
_DeepCore_ NN developing (python script, based on Keras-Tensorflow).
* toyNN : first toy for DeepCore preliminary studies, without CMSSW input
* yoloJet : full deeepCore NN developing, before integration in CMSSW
* NNPixSeed : updated version of DeepCore, with input from CMSSW and in-developing features
  * feature/new_par_try branch : old (merged) branch with new parametrization
  * EndCap branch : branch for endcap integration
 
 
## Missing directory (must be pushed in the future): 
- [] trackjet/loss_plot  
- [] trackjet/plot_performance_CMSS
- [] trackjet/keras_to_TF
- [] trackjet/Endcap_integration
- [] pdf_peak_shift : the entire analysis
- [] other missing stuff?

-->
