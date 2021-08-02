# DeepCore

This repository contains the code related to DeepCore (the CNN-based approach for the seeding in high energy jet tracking, in CMS reconstruction), outside of CMSSW. Is mostrly related to the training step, and dedicated validation plotter.

More information about DeepCore can be found at: https://twiki.cern.ch/twiki/bin/view/CMSPublic/NNJetCoreAtCtD2019

This repository contains the following directories:

## training 
It contains the script `DeepCore.py` which is the Neural Network itself. Is primary purpose is to train the model which will be used in CMSSW. The details of the usage and the options are described in the comments inside the script.
- input: root file produced using the DeepCoreNtuplizer in CMSSW (from this branch: https://github.com/vberta/cmssw/tree/CMSSW_12_0_0_pre4_DeepCoreTraining). Can be used a local input (`--input` argument) or the full statistic ntuple (hardcoded in the script) produced with the full statistic centrally produced sample. 
- training: `--training` argument performs the training, given the input
   - performed over the local input (if `--input` is used) or the central input.
   - Can be performed in multiple steps using the option `--continueTraining` from a previously produded training.
   - Epochs and input (in case of `--continueTraining`) are hardcoded, must be set in the script.
   - The details of the barrel training used in the integrated result are provided within the `DeepCore.py` script. 
   - Produces the `loss_file*.pdf` file, with the loss evolution.
   - Strongly suggested to use GPU
   - ROOT not required for this step
   - return two files: `DeepCore_train*.h5` (the weights to do prediction and so on) and `DeepCore_model*.h5` (the full model needed for CMSSW)
- prediction: `--predict` argument performs the prediction on the provided `--input`
   - if used together with `--training`  the prediction will be performed on the same sample
   - if `--input` is missing the prediction is performed on the centrally proded input
   - return `DeepCore_prediction*.npz`
- output:  `--output` argument perform validations on the prediction
   -  produces dedicated plots
   -  store the results in `DeepCore_mapValidation*`.root and `parameter_file*.pdf`

### Extra - the ntuplizer
The ntuplizer is a module of CMSSW, and build the proper input for the training of DeepCore. 
- it is contained in this branch https://github.com/vberta/cmssw/tree/CMSSW_12_0_0_pre4_DeepCoreTraining
- to obtain the ntuple two steps are needed (respective scripts contained in the `test` directory):
   1. `test_DeepCorePrepareInput.py` uses the two-file solution to combine AODSIM and GEN-SIM information and obtain a single .root file
   2. `test_DeepCoreNtuplizer.py` uses the file produced in the step 1 to build the ntuple
- the centrally produced samples are (2017 conditions, used in the integrated training):
  - barrel AOD: `/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17DRPremix-92X_upgrade2017_realistic_v10-v5/AODSIM`
  - barrel GENSIM: `/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17GS-92X_upgrade2017_realistic_v10-v1/GEN-SIM`
  - barrel prepared input (after step1): `/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/arizzi-TrainJetCoreAll-ddeeece6d9d1848c03a48f0aa2e12852/USER`
  - endcap AOD: `/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/AODSIM`
  - endcap GENSIM: `/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/GEN-SIM-DIGI-RAW`
  - endcap prepared input (after step1): `/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/vbertacc-DeepCoreTrainingSampleEC_all-3b4718db5896f716d6af32b678bbc9f2/USER`

<!---
| barrel AOD                         |`/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17DRPremix-92X_upgrade2017_realistic_v10-v5/AODSIM`|
| barrel GENSIM                      | `/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/RunIISummer17GS-92X_upgrade2017_realistic_v10-v1/GEN-SIM`|
| barrel prepared input (after step1)| `/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/arizzi-TrainJetCoreAll-ddeeece6d9d1848c03a48f0aa2e12852/USER`|
| endcap AOD                         | `/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/AODSIM`|
| endcap GENSIM                      |`/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/RunIIFall17DRStdmix-NoPU_94X_mc2017_realistic_v11-v2/GEN-SIM-DIGI-RAW`|
| endcap prepared input (after step1)| `/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/vbertacc-DeepCoreTrainingSampleEC_all-3b4718db5896f716d6af32b678bbc9f2/USER`|
--->

### Note: barrel or endcap training and status
The barrel training has been fully performed, in 2017 conditions. The endcap training is still in development (about 150 epochs on a reduced sample processed and the results are unsatisfactory).

To repeat exactly the same training as the integrated barrel only training can be obtained changing `layNum` parameter of `DeepCore.py` from 7 to 4 and use the proper input. However it should be identical to provide an input sample with 7 layers but the layers 5,6,7 empty (obtained with the `barrelTrain` argument in the ntuplizer without changing the `layNum`.)

## keras_to_TF
It contains the script `keras_to_tensorflow_custom.py`, which convert the `.h5` model returned by the `DeepCore.py --training` step to a `.pb` model, used in CMSSW. Details in the documentation inside the script.

## plotting_scripts
some auto-esplicative python plotting script for loss, validation and performance comparison

## data
some relevant updated data, hardcoded in the `DeepCore.py` script:
- barrel trained model (output of `DeepCore.py` --training): `DeepCore_barrel_weights.246-0.87.hdf5`
- barrel trained model (output of `keras_to_tensorflow_custom.py`): `DeepCoreSeedGenerator_TrainedModel_barrel_2017_246ep.pb`
- endcap weights after 150 epochs: `DeepCore_ENDCAP_train_ep150.h5`

## old development
Old development of DeepCore, kept for backup, but completely deprecated. ___Do Not Use!!!___
- `toyNN.py` : first toy for DeepCore preliminary studies, without CMSSW input
- `yoloJet.py` : full deeepCore NN developing, before integration in CMSSW
- `NNPixSeed_yolo.py` : uncleaned version of `DeepCore.py`
- `NNPixSeed_draw.py` : drawOnly version of the `NNPixSeed_yolo.py`




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
