import time
from skimage.io import imread
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.util import view_as_windows
from sklearn.externals import joblib
from scipy.signal import argrelmax
from scipy.stats import hmean
from skimage import img_as_ubyte
from config import *
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import warnings
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Test(object):
	def calculate_metrics( self, signal, ground_truth, mode ):
		signal_left = signal
		signal_list = list(signal_left)
		gt_left = ground_truth
		tp = 0
		for idx_gt , gt in enumerate( ground_truth ):
			for idx_s, s in enumerate( signal_list ):
				if s < ( gt + tp_tolerance ) and s > ( gt - tp_tolerance ):
					gt_left = gt_left[gt_left != gt]
					signal_left = signal_left[ signal_left != s ]
					tp += 1
					break
		fp = signal_left.shape[0]
		fn = gt_left.shape[0]

		if mode == "c" or mode =="o":
			return tp,fn,fp
		elif mode == "m":
			print "TP = %d" % tp
			print "FN = %d" % fn
			print "FP = %d" % fp
			print "-------------"

	def display_image_with_lines( self, signals, im , im_no, line_locs , gt ):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			im = img_as_ubyte(im)
		rgbArray = np.dstack([im, im, im])

		for x in line_locs:
			rgbArray[0:im.shape[0]/2, x - 2: x + 2] = [255, 255, 0]
		for g in gt:
			rgbArray[im.shape[0]/2:im.shape[0], int(g) - 2: int(g) + 2] = [0, 255, 255]

		f, (ax1, ax2) = plt.subplots(2, sharex=True)

		ax1.imshow(rgbArray,aspect='auto')
		ax1.axis('off')
		ax2.plot(signals)
		ax2.set_xlim([0,im.shape[1]])

		f.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0,wspace=0)
		plt.draw()
		plt.pause(1)
		k = raw_input("Enter one of b(-1) | n(+1) | m(+10) | v(-10) | e(exit):")

		if k == 'b':
			im_no -= 1
		elif k == 'n':
			im_no += 1
		elif k == 'm':
			im_no += 10
		elif k == 'v':
			im_no -= 10
		elif k == 'e':
			im_no = len(glob.glob1(test_ph,"*.png"))

		plt.close(f)
		return im_no

	def test( self, test_im_path , clf_type , feat_type , gt , mode, im_no, order = argrelmax_order ,  eps = 0.0000001, PCA_Enabled = False):
		# Load the respective model trained in the classification stage
		if clf_type == "LOG_REG":
			clf = joblib.load( model_path_logreg )
		elif clf_type == "SVM":
			clf = joblib.load( model_path_svm )
		elif clf_type == "ADA_BST":
			clf = joblib.load( model_path_ada )
		
		# Read the image, split into windows , initialize values
		im = imread( test_im_path )
		windows = view_as_windows( im , boundary_shape)
		num_windows = windows.shape[1]
		signals = np.zeros( ( num_windows , 1 ) )

		for i in range(0,num_windows):
			win = windows[:,i,:,:].reshape( boundary_shape )
			if feat_type == "LBP":
				lbp = local_binary_pattern( win , num_pts , radius , method = method_lbp )
				(hist , _) = np.histogram ( lbp.ravel() , bins = np.arange( 0 , num_pts + 3) , range = ( 0 , num_pts + 2 ) )
				hist = hist.astype("float")
				fd = hist / ( hist.sum() + eps )
				if PCA_Enabled:
					pca = joblib.load( pca_path_lbp )
					fd = pca.transform( fd )

				if clf_type == "LOG_REG":
					probs = clf.predict_proba( fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]
				elif clf_type == "SVM":
					signals[i] = clf.decision_function( fd.reshape( 1 , -1 ) )
				elif clf_type == "ADA_BST":
					probs = clf.predict_proba( fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]

			elif feat_type == "HOG":
				hog_fd = hog ( win , orientations , pixels_per_cell , cells_per_block , visualize , transform_sqrt )
				if PCA_Enabled:
					pca = joblib.load( pca_path_hog )
					hog_fd = pca.transform( hog_fd.reshape( 1 , -1 ) )
				if clf_type == "LOG_REG":
					probs = clf.predict_proba( hog_fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]
				elif clf_type == "SVM":
					signals[i] = clf.decision_function( hog_fd.reshape( 1 , -1 ) )
				elif clf_type == "ADA_BST":
					probs = clf.predict_proba( hog_fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]

			elif feat_type == "FUS":
				lbp = local_binary_pattern( win , num_pts , radius , method = method_lbp )
				(hist , _) = np.histogram ( lbp.ravel() , bins = np.arange( 0 , num_pts + 3) , range = ( 0 , num_pts + 2 ) )
				hist = hist.astype("float")
				fd = hist / ( hist.sum() + eps )

				if PCA_Enabled:
					pca = joblib.load( pca_path_lbp )
					fd = pca.transform( fd )

				hog_fd = hog ( win , orientations , pixels_per_cell , cells_per_block , visualize , transform_sqrt )
				if PCA_Enabled:
					pca = joblib.load( pca_path_hog )
					hog_fd = pca.transform( hog_fd.reshape( 1 , -1 ) )

				fd = np.concatenate( ( fd , hog_fd ) , axis = 1 )
				if clf_type == "LOG_REG":
					probs = clf.predict_proba( fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]
				elif clf_type == "SVM":
					signals[i] = clf.decision_function( fd.reshape( 1 , -1 ) )
				elif clf_type == "ADA_BST":
					probs = clf.predict_proba( fd.reshape( 1 , -1 ) )
					signals[i] = probs[0,1]

		# signals[signals < 0.3] = 0
		line_locs = argrelmax(signals,order=argrelmax_order)[0]
		line_locs += (boundary_shape[1] - 1) / 2

		if mode == "c":
			tp,fn,fp = self.calculate_metrics( line_locs , gt, mode )
			return tp,fn,fp
		elif mode == 'o':
			line_locs = argrelmax(signals,order=order)[0]
			line_locs += (boundary_shape[1] - 1) / 2
			tp,fn,fp = self.calculate_metrics( line_locs , gt, mode )
			return tp,fn,fp
		elif mode == "m":
			print "TP-FN-FP for image no: %d" %im_no
			self.calculate_metrics( line_locs , gt, mode )
			im_no = self.display_image_with_lines ( signals, im, im_no, line_locs , gt )

		return im_no

	def test_loop( self, im_no = 0 ):
		
		print "Options for the Test Stage:\n"
		print "1) Enter (c) to calculate precision, recall and f-measure metrics for the current configuration and dataset"
		print "2) Enter (m) to manually inspect the test images and predictions of the current configuration"
		print "3) Enter (o) to find the optimum order for peak localization that maximizes the f-measure CAUTION: This passes over the test set multiple times, therefore the process will take several minutes"
		print "Please enter your choice below"
		mode = raw_input()
		
		if mode == "c":
			total_tp = 0
			total_fn = 0
			total_fp = 0
			while im_no < len(glob.glob1(test_ph,"*.png")):
				image_name = "test_" + str( im_no ) + ".png"
				gt_name = "test_" + str( im_no ) + ".npy"
				im_path = os.path.join( test_ph , image_name )
				gt_path = os.path.join( test_gt_ph , gt_name )
				
				output_str = "Calculating TP-FN-FP for image %d/%d..." %( im_no + 1 , len( glob.glob1( test_ph , "*.png" ) ) )
				sys.stdout.write('%s\r' % output_str)
				tp,fn,fp = self.test( im_path , classifier_type , feat_type, np.load( gt_path ) , mode, im_no, PCA_Enabled = pca_flag )
				sys.stdout.flush()
				total_tp += tp
				total_fn += fn
				total_fp += fp
				im_no += 1

			recall = total_tp/float(total_tp+total_fn)
			precision = total_tp/float(total_tp+total_fp)
			print "\nTotal TP-FN-FP values and Recall/Precision using " + feat_type + " and " + classifier_type + "."
			print "TP = %d" % total_tp
			print "FN = %d" % total_fn
			print "FP = %d" % total_fp
			print "Recall = %f" % recall
			print "Precision = %f" % precision
			print "F-measure = %f" % hmean([recall,precision])
			print "-------------"

		elif mode == "m":

			while True:
				image_name = "test_" + str( im_no ) + ".png"
				gt_name = "test_" + str( im_no ) + ".npy"
				im_path = os.path.join( test_ph , image_name )
				gt_path = os.path.join( test_gt_ph , gt_name )

				im_no = self.test( im_path , classifier_type , feat_type, np.load( gt_path ), mode , im_no, PCA_Enabled = pca_flag )
				if im_no < 0 or im_no >= len(glob.glob1(test_ph,"*.png")):
					break
					
		elif mode == "o":
			order_list = range(tp_tolerance , tp_tolerance + 15)
			precision_list = []
			recall_list = []
			fmeasure_list = []
			for order in order_list:
				total_tp = 0
				total_fn = 0
				total_fp = 0
				im_no = 0
				print "Calculating metrics with peak localization order: %d" %(order)
				print "------------------------------------------------------"
				while im_no < len(glob.glob1(test_ph,"*.png")):
					image_name = "test_" + str( im_no ) + ".png"
					gt_name = "test_" + str( im_no ) + ".npy"
					im_path = os.path.join( test_ph , image_name )
					gt_path = os.path.join( test_gt_ph , gt_name )
					output_str = "Calculating TP-FN-FP for image %d/%d..." %(im_no + 1,len( glob.glob1( test_ph , "*.png" ) ) )
					sys.stdout.write('%s\r' % output_str )
					tp,fn,fp = self.test( im_path , classifier_type , feat_type, np.load( gt_path ) , mode, im_no, order = order , PCA_Enabled = pca_flag )
					sys.stdout.flush()
					total_tp += tp
					total_fn += fn
					total_fp += fp
					im_no += 1
				recall_list.append( total_tp/float(total_tp+total_fn) )
				precision_list.append( total_tp/float(total_tp+total_fp) )
				fmeasure_list.append( hmean([total_tp/float(total_tp+total_fn)  , total_tp/float(total_tp+total_fp)]) )
			
			max_fmeasure = max(fmeasure_list)
			max_index = fmeasure_list.index(max_fmeasure)
			optimum_order = order_list[max_index]
			print "Optimum order for peak localization with respect to f-measure is: %d" %( optimum_order )
			print "You may wish to change your argrelmax_order in config.cfg to %d" %( optimum_order )
			prec = plt.scatter( np.asarray( order_list ) , np.asarray( precision_list ) , c = 'm' , marker = '^' )
			rec = plt.scatter( np.asarray( order_list ) , np.asarray( recall_list ) , c = 'r' , marker = 'o' )
			fm = plt.scatter( np.asarray( order_list ) , np.asarray( fmeasure_list ) , c = 'b' , marker = 'd' )
			plt.legend((prec,rec,fm) , ("Precision" , "Recall" , "F-Measure") , scatterpoints = 1 , fontsize = 10 )
			plt.show()
			