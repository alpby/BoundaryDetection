from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import numpy as np
import glob
import os
from config import *
import time

# Classifier Class:
# With main method classify(), creates a binary classification model from previously extracted feature set
# First, it appends and labels all single feature vectortogether
# Then, it trains a classification model with respect to the input of the user
# 		clf_type: type of the classifier to be used, check kernel also for method SVM
# 		kernel: type of kernel  to be used for the SVM method
# 		feat_type: which feature type should be used as the feature set
# 		pca_enabled: flag variable to enable/disable Principal Component Analysis
		
class Classifier(object):
	def append_features_labels_pca_LBP ( self, pca_enabled_flag ):
		pos_feat_path = pos_feat_ph_lbp
		neg_feat_path = neg_feat_ph_lbp
		pca_path = pca_path_lbp
		fds = []
		labels = []
		for feat_path in glob.glob( os.path.join( pos_feat_path, "*.npy" ) ):
			fds.append( np.load( feat_path ) )
			labels.append( 1 )
		for feat_path in glob.glob( os.path.join( neg_feat_path, "*.npy" ) ):
			fds.append( np.load( feat_path ) )
			labels.append( 0 )
		if pca_enabled_flag == True:
			pca = PCA( n_components = pca_dim_lbp )
			pca.fit( fds )
			print "PCA Sum of Explained Variance Ratios = %f" %( sum ( pca.explained_variance_ratio_ ) )
			print "-----------------------------"
			fds = pca.transform( fds )
			print "Your LBP feature set has %d observations and %d features after PCA" %( fds.shape[0] , fds.shape[1] )
			print "-----------------------------"
			if not os.path.isdir( os.path.split( pca_path )[0] ):
				os.makedirs( os.path.split( pca_path )[0] )
			joblib.dump( pca , pca_path )

		return ( fds , labels )

	def append_features_labels_pca_HOG ( self, pca_enabled_flag ):
		pos_feat_path = pos_feat_ph_hog
		neg_feat_path = neg_feat_ph_hog
		pca_path = pca_path_hog
		fds = []
		labels = []
		for feat_path in glob.glob( os.path.join( pos_feat_path, "*.npy" ) ):
			fds.append( np.load( feat_path ) )
			labels.append( 1 )
		for feat_path in glob.glob( os.path.join( neg_feat_path, "*.npy" ) ):
			fds.append( np.load( feat_path ) )
			labels.append( 0 )
		if pca_enabled_flag == True:
			pca = PCA( n_components = pca_dim_hog )
			pca.fit( fds )
			print "PCA Sum of Explained Variance Ratios = %f" %( sum ( pca.explained_variance_ratio_ ) )
			print "-----------------------------"
			fds = pca.transform( fds )
			print "Your HOG feature set has %d observations and %d features after PCA" %( fds.shape[0] , fds.shape[1] )
			print "-----------------------------"
			if not os.path.isdir( os.path.split( pca_path )[0] ):
				os.makedirs( os.path.split( pca_path )[0] )
			joblib.dump( pca , pca_path )

		return ( fds , labels )

	def classify( self, clf_type , feat_type , kernel , PCA_Enabled = False ):

		if feat_type == "LBP":
			( fds , labels ) = self.append_features_labels_pca_LBP( PCA_Enabled )
		elif feat_type == "HOG":
			( fds , labels ) = self.append_features_labels_pca_HOG( PCA_Enabled )
		elif feat_type == "FUS":
			# List LBP features and apply PCA if requested
			( fds_lbp , labels ) = self.append_features_labels_pca_LBP( PCA_Enabled )

			# List HOG features and apply PCA if requested
			( fds_hog , _ ) = self.append_features_labels_pca_HOG( PCA_Enabled )

			# Concatenate two feature set together
			fds_lbp = np.vstack ( fds_lbp )
			fds_hog = np.vstack ( fds_hog )
			fds = np.concatenate( ( fds_lbp , fds_hog ) , axis = 1 )
			labels = np.vstack( labels ).ravel()
			del fds_lbp , fds_hog

		# Fit model wrt to the chosen classifier
		if clf_type == "SVM":
			clf = SVC(kernel = kernel)
			print "Training SVM Classifier with kernel: %s ..." %(kernel)
			clf.fit( fds , labels )
			model_path = model_path_svm
			print "SVM Score %f:" %( clf.score( fds , labels ) )
			print "------------------------------"
		elif clf_type == "LOG_REG":
			clf = LogisticRegression()
			print "Training Logistic Regression ..."
			clf.fit( fds , labels )
			model_path = model_path_logreg
			print "Logistic Regression Score: %f" %( clf.score( fds , labels ) )
			print "------------------------------"
		elif clf_type == "ADA_BST":
			clf = AdaBoostClassifier( DecisionTreeClassifier( max_depth = 1 ), algorithm = "SAMME")
			print "Training AdaBoost with Decision Stumps ..."
			clf.fit( fds , labels )
			print "AdaBoost Score: %f" %( clf.score( fds , labels ) )
			print "---------------"
			model_path = model_path_ada
		else:
			print "Classification model type is unknown!"


		# Save the model for later use
		if not os.path.isdir( os.path.split( model_path )[0] ):
			os.makedirs( os.path.split( model_path )[0] )
		joblib.dump( clf , model_path )
		print "Model Trained!"
