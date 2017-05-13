import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import *

class ScreePlot(object):
	def pca_analysis( self , feat_type ):
		
		if feat_type == "LBP" or feat_type == "FUS":
			print "Please wait while the screeplot for LBP feature is being prepared..."
			print "Please observe the plot carefully and try to detect after how many dimensions the marginal change in explained variance becomes negligibe"
			print "It is advised to choose that number as the related pca_dim in config.cfg for the rest of your analysis"
			print "------------------------------------------------------------------------------------------------------"
			pos_feat_path = pos_feat_ph_lbp
			neg_feat_path = neg_feat_ph_lbp
		
			pos_fds = []
			neg_fds = []
			for feat_path in glob.glob( os.path.join( pos_feat_path, "*.npy" ) ):
				fd = np.load( feat_path )
				pos_fds.append( fd )
			
			for feat_path in glob.glob( os.path.join( neg_feat_path, "*.npy" ) ):
				neg_fds.append( np.load( feat_path ) )
			
			pos_fds = np.vstack( pos_fds )
			neg_fds = np.vstack( neg_fds )
			fds = np.concatenate( ( pos_fds , neg_fds ) , axis = 0 )
			self.plot_scree( fds , "LBP" )
	
		if feat_type == "HOG" or feat_type == "FUS":
			print "Please wait while the screeplot for HOG feature is being prepared..."
			print "Please observe the plot carefully and try to detect after how many dimensions the marginal change in explained variance becomes negligibe"
			print "It is advised to choose that number as the related pca_dim in config.cfg for the rest of your analysis"
			print "------------------------------------------------------------------------------------------------------"
			pos_feat_path = pos_feat_ph_hog
			neg_feat_path = neg_feat_ph_hog
			
			pos_fds = []
			neg_fds = []
			for feat_path in glob.glob( os.path.join( pos_feat_path, "*.npy" ) ):
				fd = np.load( feat_path )
				pos_fds.append( fd )
			
			for feat_path in glob.glob( os.path.join( neg_feat_path, "*.npy" ) ):
				neg_fds.append( np.load( feat_path ) )
			
			pos_fds = np.vstack( pos_fds )
			neg_fds = np.vstack( neg_fds )
			fds = np.concatenate( ( pos_fds , neg_fds ) , axis = 0 )
			self.plot_scree( fds , "HOG" )
		
	def plot_scree( self , data , feat ):
		pca = PCA( )
		pca.fit( data )
		
		plt.scatter( 1 + np.arange( len( pca.explained_variance_ratio_ ) )  , pca.explained_variance_ratio_ , c ='r' , marker = 'o' )
		plt.title( 'Scree Plot for ' + feat)
		plt.show()
		
