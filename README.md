# cms_mywork
This is the area where I store all the code developed outside shared repositories of CMS. It contains quite unorganized software that will be eventually included in CMSSW or others collaboration frameworks in the future. 

A raw description of the directory tree (or particular branches):


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
