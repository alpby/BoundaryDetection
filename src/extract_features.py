from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
import numpy as np
from config import *
import time
import sys

# Features Class:
# Collects the paths of the source images and target path to extract features to with respect to the feature type chosen
		# im_paths: Where to look for sourcimages
		# feat_ph_hog: Where to extract HOG feature to
		# feat_ph_lbp: Where to extract LBP features to
		# des_type: Type of the feature descriptor chosen
		# neg_flag: flow control variable
		# hog_check: flow control variable
		# lbp_check: flow controvariable
		# eps: extremely small number to avoid division by zero

class Features(object):
	def extract_features( self, im_paths , feat_ph_hog , feat_ph_lbp , des_type , neg_flag , hog_check , lbp_check , eps = 0.0000001 ):
		if neg_flag:
			feat_pos_neg = "negative"
		else:
			feat_pos_neg = "positive"
			
		if  ( hog_check & ( des_type == "HOG" or des_type == "FUS" ) ):
			print "Calculating the HOG descriptors for %s samples and saving them..." %(feat_pos_neg)
			paths = glob.glob1( im_paths , "*.png" )
			for idx, im_path in enumerate(paths):
				sys.stdout.write('Done: %d/%d\r'%( idx , len(paths) ) ) 
				im = imread( os.path.join ( im_paths , im_path )  , as_grey = True )
				fd = hog ( im , orientations , pixels_per_cell , cells_per_block , visualize , transform_sqrt )
				feat_ph = feat_ph_hog
				if not fd.shape[0] == 0:
					fd_name = os.path.split( im_path )[1].split(".")[0] + ".npy"
					fd_path = os.path.join( feat_ph , fd_name )
					np.save( fd_path , fd )
				else:
					print "fd shape is 0"
				sys.stdout.flush()
			print "Features saved in %s \n" %(feat_ph_hog)
		
		if ( lbp_check & ( des_type == "LBP" or des_type == "FUS" ) ):
			print "Calculating the LBP descriptors for %s samples and saving them..." %(feat_pos_neg)
			paths = glob.glob1( im_paths , "*.png" )
			for idx, im_path in enumerate( paths ):
				sys.stdout.write('Done: %d/%d\r'%( idx , len(paths) ) )
				im = imread( os.path.join ( im_paths , im_path )  , as_grey = True )
				lbp = local_binary_pattern( im , num_pts , radius , method = method_lbp )
				(hist , _) = np.histogram ( lbp.ravel() , bins = np.arange( 0 , num_pts + 3) , range = ( 0 , num_pts + 2 ) )
				hist = hist.astype("float")
				fd = hist / ( hist.sum() + eps )
				feat_ph = feat_ph_lbp
				if not fd.shape[0] == 0:
					fd_name = os.path.split( im_path )[1].split(".")[0] + ".npy"
					fd_path = os.path.join( feat_ph , fd_name )
					np.save( fd_path , fd )
				else:
					print "fd shape is 0"
				sys.stdout.flush()
			print "Features saved in %s \n" %(feat_ph_lbp)
		
		if lbp_check & neg_flag & ( des_type == "LBP" or des_type == "FUS" ):
			log = {'num_pts': num_pts , 'radius':radius , 'method_lbp': method_lbp}
			with open( lbp_log_ph , 'w' ) as outfile:
				json.dump( log , outfile )
				print "LBP log file created"
				
		if hog_check & neg_flag & ( des_type == "HOG" or des_type == "FUS" ):
			log = {'orientations': orientations , 'pixels_per_cell':pixels_per_cell , 'cells_per_block': cells_per_block,
			'visualize':visualize , 'transform_sqrt':transform_sqrt}
			with open( hog_log_ph , 'w' ) as outfile:
				json.dump( log , outfile )
				print "HOG log file created"
	

	

		
