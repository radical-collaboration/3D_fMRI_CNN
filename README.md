# 3D_fMRI_CNN

[Experiments](https://docs.google.com/spreadsheets/d/1zCSbVuOSFjKyq8pNWNHabra7zJJdAljbujWsr5N4szs/edit?usp=sharing)

This repository is curated by members of the RADICAL team at Rutgers University, Pouya Bashivan (MIT), Irina Rish (IBM), and Mina Gheiratmand (University of Alberta) 

The objective is to improve classification accuracy in schizophrenia and other cognitive disorders using fMRI and 3D recurrent ConvNet. This will be done by refining to the underlying algorithms, converting fMRI data into split 3-dimensional movies, identifying and increasing parallelism in existing code, and extending the single machine code to run on GPU HPCs

Requirements: 

python libraries are included in requirements.txt

cuDNN/5.0-CUDA-7.5.18

Theano/0.8.2-Python-2.7.9-noMPI configured to run on GPU to enable Conv3DDNNLayer from lasagne.layers.dnn

Command: 

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn_fmri.py --num_epochs 10 --batch_size 30 --num_folds 10 --num_classes 2 --grad_clip 100 --model 'mix' --num_input_channels 1

