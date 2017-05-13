# Import modules
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import glob
import os
from skimage.io import imread

from config import *

import numpy as np
import scipy.misc

from scipy.stats import describe
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def scree_plot ( feat_mat ):
	num_vars = feat_mat.shape[1]
	U, S, V = np.linalg.svd(feat_mat)
	eigvals = S**2 / np.cumsum(S)[-1]

	fig = plt.figure(figsize=(8,5))
	sing_vals = np.arange(num_vars) + 1
	plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
	plt.title('Scree Plot')
	plt.xlabel('Principal Component')
	plt.ylabel('Eigenvalue')
	#I don't like the default legend so I typically make mine like below, e.g.
	#with smaller fonts and a bit transparent so I do not cover up data, and make
	#it moveable by the viewer in case upper-right is a bad place for it
	leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
					 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
					 markerscale=0.4)
	leg.get_frame().set_alpha(0.4)
	leg.draggable(state=True)
	plt.show()

def save_feature_arrays(feat_type):
	if feat_type == "LBP":
		pos_feat_path = pos_feat_ph_lbp
		neg_feat_path = neg_feat_ph_lbp
	elif feat_type == "HOG":
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
	scree_plot( pos_fds )
	scree_plot( neg_fds )
	np.save( "pos_fds.npy" , pos_fds )
	np.save( "neg_fds.npy" , neg_fds )

def box_plot_features():
	pos_fds = np.load( "pos_fds.npy" )
	neg_fds = np.load( "neg_fds.npy" )

	plt.figure()
	plt.boxplot(pos_fds)
	plt.show()

	plt.figure()
	plt.boxplot(neg_fds)
	plt.show()

def surf_plot( image_path ):
	im = imread( image_path )
	# im = scipy.misc.imresize( im , 0.8 , interp = 'cubic')
	xx, yy = np.mgrid[ 0:im.shape[0], 0:im.shape[1] ]
	fig = plt.figure()
	ax = fig.gca( projection='3d' )
	ax.plot_surface(xx, yy, im ,rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
	plt.show()



# neg_im_path = "C:/Users/user/Documents/EE475/Proje/data/vispera_boundary_dataset_20160526/train_neg/neg_1_3_152.png"
# pos_im_path = "C:/Users/user/Documents/EE475/Proje/data/vispera_boundary_dataset_20160526/train_pos/pos_10_5.png"
# surf_plot(neg_im_path)

save_feature_arrays("HOG")
# feat_analysis()
