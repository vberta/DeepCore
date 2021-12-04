#!/bin/bash

cd work/DeepCore/CMSSW_10_2_5/src/DeepCore/Training1204/
pip install uproot3
python ../training/DeepCore_GPU.py --training --input DeepCoreTrainingSample.root
