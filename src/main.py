from classifier import *
from extract_features import *
from test import *
from config import *
from scree_plot import *
import time
import glob
import os
import json

# Main script of the program, consists of 2 parts:
# 1) Extracting features from positive and negative samples:
# 			feat_type: the type of feature one would like to extract, possible options "HOG", "LBP" or "FUS", make changes in config.cfg
# Optional Stage: Run PCA Analysis 
# 2) Classifier training using the features extracted in step 1
# 			classifier_type: Classifier type to be used for binary classification, possible options 
# 				LOG_REG: logistic Regression
# 				SVM    : Support Vector Machines
# 				ADA_BST: AdaBoost  
# 			pca_flag       : if True, Principal Component Analysis is applied to related feature set with pca_dim_hog for HOG and pca_dim_lbp for LBP
# 3) Testing the binary classifier trained in step 2 by using sliding window and peak localization techniques
	
def create_dirs():
	for dir in [pos_feat_ph_hog , pos_feat_ph_lbp, neg_feat_ph_hog, neg_feat_ph_lbp, model_path]:
		if not os.path.exists( dir ):
			os.makedirs( dir )

def check_lbp_log( ):
	if os.path.exists( lbp_log_ph ):
		with open( lbp_log_ph ) as data:
			log = json.load( data )
		if all( [ log["num_pts"] == num_pts , log["radius"] == radius , log["method_lbp"] == method_lbp ] ):
			print "Previous LBP feature extraction configuration matches your current LBP feature configuration"
			print "If you haven't changed your dataset, you may wish to skip LBP feature extraction"
			s = raw_input("Enter (y) if you wish to skip LBP feature extraction. Enter any other key otherwise.\n")
			print "--------------------------------------------------------------------------------------------"
			if s == 'y':
				return False
			else:
				os.remove( lbp_log_ph )
				for dir in [pos_feat_ph_lbp,neg_feat_ph_lbp]:
					for f in glob.glob1(dir,"*.npy"):
						os.remove( os.path.join( dir , f ) )
				return True
	else:
		for dir in [pos_feat_ph_lbp,neg_feat_ph_lbp]:
			for f in glob.glob1(dir,"*.npy"):
				os.remove( os.path.join( dir , f ) )
		return True

def check_hog_log( ):
	if os.path.exists( hog_log_ph ):
		with open( hog_log_ph ) as data:
			log = json.load( data )
		if all( [ log["orientations"] == orientations , log["pixels_per_cell"] == pixels_per_cell , 
		log["cells_per_block"] == cells_per_block, log["visualize"] == visualize , log["transform_sqrt"] == transform_sqrt ] ):
			print "Previous HOG feature extraction configuration matches your current HOG feature configuration"
			print "If you haven't changed your dataset, you may wish to skip HOG feature extraction"
			s = raw_input("Enter (y) if you wish to skip HOG feature extraction. Enter any other key otherwise.\n")
			print "--------------------------------------------------------------------------------------------"
			if s == 'y':
				return False
			else:
				os.remove( hog_log_ph )
				for dir in [pos_feat_ph_hog,neg_feat_ph_hog]:
					for f in glob.glob1(dir,"*.npy"):
						os.remove( os.path.join( dir , f ) )
				return True
	else:
		for dir in [pos_feat_ph_hog,neg_feat_ph_hog]:
			for f in glob.glob1(dir,"*.npy"):
				os.remove( os.path.join( dir , f ) )
		return True

def check_feat_type():
	if not ( feat_type in ["FUS","HOG","LBP"] ):
		raise ValueError("Feature type chosen is not applicable")

def check_classifier_type():
	if not ( classifier_type in ["LOG_REG","ADA_BST","SVM"]):
		raise ValueError("Classifier type chosen is not applicable")
	elif ( classifier_type == "SVM" ) & ( not ( svm_kernel in ['rbf','linear','poly','sigmoid'] ) ):
		raise ValueError("SVM Kernel type is not supported")

def main():
	create_dirs()
	check_feat_type()
	check_classifier_type()
	
	
	print "-------------------------------------------------------------------------"
	print "\n"	
	print "                      FEATURE EXTRACTION STAGE                           "
	print "\n"
	print "-------------------------------------------------------------------------"
	
	hog_check = check_hog_log()
	lbp_check = check_lbp_log()

	Features().extract_features( train_pos_ph , pos_feat_ph_hog , pos_feat_ph_lbp , des_type = feat_type , 
	neg_flag = False , hog_check = hog_check, lbp_check = lbp_check)
	Features().extract_features( train_neg_ph , neg_feat_ph_hog , neg_feat_ph_lbp , des_type = feat_type ,
	neg_flag = True , hog_check = hog_check, lbp_check = lbp_check)
	print "Would you like to run a PCA analysis before proceeding to classification stage?"
	s = raw_input("Enter (y) to proceed with PCA analysis, enter any other key otherwise \n")
	if s == 'y':
		ScreePlot().pca_analysis( feat_type )
	
	print "-------------------------------------------------------------------------"
	print "\n"	
	print "                       CLASSIFICATION STAGE                              "
	print "\n"
	print "-------------------------------------------------------------------------"
	filelist = glob.glob1(model_path,"*.model")    
	for f in filelist:
		os.remove(model_path + "/" + f)
	
	Classifier().classify( clf_type = classifier_type , feat_type = feat_type, PCA_Enabled = pca_flag , kernel = svm_kernel)
	
	print "-------------------------------------------------------------------------"
	print "\n"	
	print "                            TEST STAGE                                   "
	print "\n"
	print "-------------------------------------------------------------------------"
	Test().test_loop()

if __name__ == '__main__':
    tick = time.clock()
    main()
    tock = time.clock()
    print "Elapsed Time:"
    print tock - tick
