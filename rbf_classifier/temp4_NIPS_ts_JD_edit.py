# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:33:24 2016

@author: pipolose
"""

import polyssifier_21_4_16 as ps
import logging
from sklearn.cross_validation import KFold
from os import path
import h5py 
import numpy as np


def load_data():
  """
  Loads the data from nii files.

  Parameters
  ----------
  data_file: str

  Returns
  -------
  data: array_like
  """
  ##### Load labels
  f = h5py.File(
     '/Users/JumanaDakka/git/new_3D_fmri_CNN_for_sftp/3D_fMRI_CNN/rbf_classifier/shuffled_output_runs.hdf5', 'r')
  g = h5py.File(
     '/Users/JumanaDakka/git/new_3D_fmri_CNN_for_sftp/3D_fMRI_CNN/rbf_classifier/shuffled_output_labels.hdf5', 'r')
  h = h5py.File(
     '/Users/JumanaDakka/git/new_3D_fmri_CNN_for_sftp/3D_fMRI_CNN/rbf_classifier/shuffled_output_subjects.hdf5', 'r')
  i = h5py.File(
     '/Users/JumanaDakka/git/new_3D_fmri_CNN_for_sftp/3D_fMRI_CNN/rbf_classifier/shuffled_output_features.hdf5', 'r')

  #f = h5py.File(
  #  '/braintree/data2/active/users/bashivan/Data/fmri_conv_orig/shuffled_output_runs.hdf5', 'r')
  #g = h5py.File(
  #  '/braintree/data2/active/users/bashivan/Data/fmri_conv_orig/shuffled_output_labels.hdf5', 'r')
  #h = h5py.File(
  #  '/braintree/data2/active/users/bashivan/Data/fmri_conv_orig/shuffled_output_subjects.hdf5', 'r')
  #i = h5py.File(
  #  '/braintree/data2/active/users/bashivan/Data/fmri_conv_orig/shuffled_output_features.hdf5', 'r')
  subjects, labels, features, runs = [], [], [], []

  subjects = h['subjects'][()]
  labels = g['labels'][()]
  runs = f['runs'][()]
  features = i['features'][()]


  print features.shape
  # Load features
  features = np.expand_dims(np.array(features).transpose([0, 4, 3, 1, 2]),
                            axis=2)  # Add another filler dimension for the samples
  # collect sites

  
  print features.shape
  
  # change labels from -1/1 to 0/1
  labels = (np.array(labels, dtype=int) == 1).astype(int)
  # labels[:10] = 1
  labels = [int(i) for i in labels]
  labels = np.asarray(labels)

  # change subject_IDs to scale 0-94
  unique_IDs = []
  [unique_IDs.append(i) for i in subjects if not unique_IDs.count(i)]
  dictionary_IDs = {x: i for i, x in enumerate(unique_IDs, start=1)}

  for i in range(len(subjects)):
    subjects[i] = dictionary_IDs[subjects[i]]

  subjects = np.asarray(subjects, dtype=int)
  
  
  return features, labels, subjects, np.asarray(runs)



#import ipdb

'''
INPUT PARAMETERS
'''

#mname = ('/home/mgheirat/breezy_migration/denoised_fBIRN_AudOdd_allsites_0003_standardized_3stage_Global_ts.mat')
#         'denoised_fBIRN_AudOdd_allsites_0003_sitestdzd_corr_subset_v2_sqz.mat')
#out_dir = '/home/mgheirat/SrgyClassifiers_95fold_Sep25_LPf_denoised_sitestdzd_all_corr'
#out_dir = '/home/mgheirat/SrgyClassifiers_95fold_Jan31_sitestdzd_all_corr_NN_32'

out_dir = '/Users/JumanaDakka/git/new_3D_fmri_CNN_for_sftp/3D_fMRI_CNN/rbf_classifier'

#ksplit = 5  

### in matlab code: 
#cd.folds = size(data,1)/4
#n = 0:log2(size(data,2)) ;
#cd.top_K =  [ 2.^n ,  size(data,2)-1 ];
#numTopVars = [1	,2,4,8,16,32,64,128,256,512,	1024,	2048,	4096,	8192,	16384, 32768,	65536,	131072, 161596]

#numTopVars = [ 1,  4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 81924, 16384, 32768,  65536,  100000 , 161596]
#numTopVars= [1, 3, 5, 10, 30, 32, 50,  100, 200, 300, 500, 700, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 100000, 150000]


numTopVars =[77953] # [32]
NAMES = ["Chance", "RBF SVM" ] #[Naive Bayes", "Linear SVM"]

#NAMES = ["Chance", "Nearest Neighbors", "Linear SVM", "RBF SVM",
#          "Decision Tree", "Random Forest", "Logistic Regression",
#          "Naive Bayes", "LDA"]
#NAMES = [ "Nearest Neighbors", "Linear SVM",
#          "Decision Tree", "Random Forest",
#          "Naive Bayes","Logistic Regression", "LDA", "RBF SVM"]

#NAMES = ["Chance", "Nearest Neighbors", "Linear SVM", "Decision Tree",
#         "Logistic Regression", "Naive Bayes", "LDA"]
# NAMES = ["Random Forest"]
          
#NAMES = [ "Logistic Regression", "Decision Tree","Random Forest",
#         "Naive Bayes", "LDA"]
         
if __name__ == "__main__":
    '''
    Initializing logger to write to file and stdout
    '''
    logging.basicConfig(format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(out_dir, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    '''
    DATA LOADING
    '''

    data, labels, subjects, runs = load_data()
    fold_pairs = []
    num_folds = 5
    print num_folds

    sub_nums = subjects
    subs_in_fold = np.ceil(np.max(sub_nums) / float(num_folds))

    # n-fold cross validation
    #for i in range(num_folds):
        
       # test_ids = np.bitwise_and(sub_nums >= subs_in_fold * (i), sub_nums < subs_in_fold * (i + 1))
       # train_ids = ~ test_ids
        #fold_pairs.append((np.nonzero(train_ids)[0], np.nonzero(test_ids)[0]))

    
    #data, labels, data_file = ps.load_data(mname)
    
    #for foldNum, fold in enumerate(fold_pairs):
        #print data[:, fold[1]], np.squeeze(labels[fold[1]])
        #print data[:,fold[1]]
    
    #, self.labels[fp[0]]
    classifiers, params = ps.make_classifiers(NAMES)  # data.shape, ksplit)  # Check

    
    #kf = KFold(labels.shape[0], n_folds=ksplit)
    
    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    
    #fold_pairs = [(tr, ts) for (tr, ts) in kf]
    
    ksplit = 5
    #assert len(fold_pairs) == ksplit
    
    '''
    RANK VARIABLES FOR EACH FOLD (does ttest unless otherwise specified)
    '''
    kf = KFold(labels.shape[0], n_folds=ksplit)
    fold_pairs = [(tr, ts) for (tr, ts) in kf]

    #print fold_pairs
    rank_per_fold = ps.get_rank_per_fold(data, labels, fold_pairs,
                                         save_path=out_dir, parallel=True)
    
    '''
    COMPUTE SCORES
    '''
    
    score={}
    dscore=[]   
    totalErrs = []
    
    for name in NAMES:
        mdl = classifiers[name]
        param = params[name]
        # get_score runs the classifier on each fold, each subset of selected top variables and does a grid search for classifier-specific parameters (selects the best)
        clf, allConfMats, allTotalErrs = ps.get_score(data, labels,
                                                      fold_pairs, name,
                                                      mdl, param,
                                                      numTopVars=numTopVars,
                                                      rank_per_fold=rank_per_fold,
                                                      parallel=True,
                                                      rand_iter=10)
        # was 10. I think the final  results were gained with 10
        # save classifier object and results to file
        ps.save_classifier_results(name, out_dir, allConfMats, allTotalErrs)
        ps.save_classifier_object(clf, name, out_dir)
        # Append classifier results to list of all results
        dscore.append(allConfMats)
        totalErrs.append(allTotalErrs)
    
    '''
    First do some saving of total results
    '''
    filename_base = path.splitext(path.basename(mname))[0]
    ps.save_combined_results(NAMES, dscore, totalErrs,
                             numTopVars, out_dir, filename_base)
        #Now comes the plotting!!
    ps.plot_errors(NAMES,numTopVars, dscore, totalErrs, 
                    filename_base, out_dir)
    
    logging.shutdown()
