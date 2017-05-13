import os
import glob
import numpy as np
import random
import scipy.io as sio
from skimage import *
from skimage import io, filters, exposure, transform
from config import *
import warnings
import time
import json
import sys

# Script for dataset creation. Check [dataset] part of config.cfg to configure parameters 
def create_dirs():
	for dir in [train_pos_ph, train_neg_ph, test_ph, test_gt_ph]:
		if not os.path.exists(dir):
			os.makedirs(dir)

def check_dataset_log():
	if os.path.exists( dataset_log_ph ):
		with open( dataset_log_ph ) as data_file:    
			log = json.load( data_file )
		if all( [ log["train_ratio"] == train_ratio , log["boundary_shape"] == boundary_shape , log["sampling_seed"] == sampling_seed , 
		log["num_neg_sample_per_image"] == num_neg_sample_per_image ] ):
			print 'Previous dataset creation configuration matches your current configuration'
			s = raw_input( 'Enter (y) if you wish to skip dataset creation process. Enter any other key to proceed with dataset creation\n')
			if s == 'y':
				return True
			else:
				os.remove( dataset_log_ph )
				return False
	else:
		return False

tick = time.clock()
# ---------------------------------------------     PART (0)    ----------------------------------------------- #
# Throw preliminary exceptions
if train_ratio >= 1 or train_ratio <= 0:
	raise ValueError('Training sample ratio is not between 0 and 1')
if boundary_shape[1] % 2 == 0:
	raise ValueError('Please enter an odd boundary width (boundary_shape[1] in config.cfg')
# Create directories if they don't exist
create_dirs()
# Dataset creation can be skipped by the will of the user
if check_dataset_log():
	import sys
	sys.exit( "Dataset creation aborted" )


# ---------------------------------------------     PART (1)    ----------------------------------------------- #  
# Remove previously created positive and negative train samples, test samples, rescaled test sample annotations, dataset log path #

# (1a) Delete images files in related test and train paths
for dir in [train_pos_ph,train_neg_ph,test_ph]:
    filelist = glob.glob1( dir , "*.png" )
    for f in filelist:
        os.remove( os.path.join( dir , f ) )
# (1b) Delete column annotations for test images
filelist = glob.glob1(test_gt_ph,"*.npy")
for f in filelist:
    os.remove( os.path.join( test_gt_ph , f ) )
	

# ---------------------------------------------     PART (2)    ----------------------------------------------- # 
#          Create new positive and negative train samples, test samples, rescaled test sample annotations       #

# (2a) 
# Partition original images into test and training set 
# training_ratio: Defines what fraction of the whole original images should be partitioned as training images
# sampling_seed : Fixed in order to reproduce the same train and test sets each run of dataset.py
# boundary_shape: Defines the fixed window size to be used 
#                 boundary_shape[0]: Fixed height for all positive and negative samples and rescaled test images
#                 boundary_shape[1]: Fixed width for all positive and negative samples
print "Creating positive and negative samples with boundary shape:%d x %d...\n" %(boundary_shape[0],boundary_shape[1])
# os.chdir(data_ph) !!!
warnings.filterwarnings( "ignore" )
listing = glob.glob1( images_ph, "*.jpg" )
num_images = len( listing )
num_train = int( round( num_images * train_ratio ) )
random.seed( sampling_seed )
# Sample training list from the original set of images
train_listing = random.sample( listing , num_train )
num_train = len( train_listing )
# Test list is whatever is left from the training list
test_listing = list( set( listing ) - set( train_listing ) )

# (2b) 
# Preprocess and rescale original images, crop positive and negative samples out of them
# Shelf images are preprocessed and brought to a common height ( boundary_shape[0] )
# Then they are cropped from the annotated positions, with the width defined by boundary_shape[1]
# Cropped parts are stored as positive samples, then the rest of the image is sampled from random positions and cropped at the same size
# 				num_neg_sample_per_image: number random points chosen as the center of the non-boundary object

for i,image_path in enumerate(train_listing):
	sys.stdout.write('Done: %d/%d\r'%( i , len( train_listing ) ) )
	im = img_as_ubyte(io.imread( os.path.join( images_ph , image_path ), as_grey = True) )
	image_name = os.path.splitext( image_path )[0]
	cols = np.ravel(sio.loadmat( os.path.join( annotations_ph , image_name + '.mat') )['cols'] )
	cols -= 1                                       # Conversion from MATLAB indexing to Python indexing
	( ymax , _ ) = im.shape
	rescale_ratio = float( boundary_shape[0] ) / float( ymax )        # Calculate the rescale ratio to scale the image height to common height	
	cols = cols * rescale_ratio          # Rescale the columns
	im = transform.rescale( im , rescale_ratio )    # Rescale the image  
	im = filters.gaussian( im , sigma = 1)          # Apply Gaussian filter
	im = exposure.equalize_hist( im )               # Histogram equalization to eliminate illumination differences
	( _ , xmax ) = im.shape
	boundary_width = ( boundary_shape[1] - 1 ) / 2  # Number of pixels to be included to the left and rigth of the column annotation
	im_original = im                                # Store im_original to be used by column annotations
	# Crop positive samples and remove them from the original image	
	for idx, x in enumerate( np.nditer( cols ) ):
		x = int( x )
		if min( x + boundary_width , xmax ) - max( x - boundary_width , 0 ) == 2 * boundary_width:
			slice = np.arange( ( x - boundary_width ) , ( x + boundary_width + 1 ) )
			cropped_object = im_original[ 0:ymax , slice ]
			io.imsave( os.path.join( train_pos_ph , "pos" + str(i) + "_" + str(idx) + ".png" ) , cropped_object )
			im = np.delete( im , slice , axis = 1 )
	( _ , xmax ) = im.shape
	# Select random locations from the rest of the image
	if len( range( boundary_width + 1 , xmax - boundary_width ) ) < num_neg_sample_per_image:
		num_possible_samples = len( range( boundary_width + 1 , xmax - boundary_width ) )
	else:
		num_possible_samples = num_neg_sample_per_image
		
	neg_locations = random.sample( range( boundary_width + 1 , xmax - boundary_width ) , num_possible_samples )
	for idx, n in enumerate( neg_locations ):
		slice = np.arange( ( n - boundary_width ) , ( n + boundary_width + 1 ) )
		cropped_object = im[ 0:ymax , slice ]
		io.imsave( os.path.join ( train_neg_ph , "neg" + str(i) + "_" + str(idx) + ".png" ) , cropped_object )
	sys.stdout.flush()

print "Created %d positive and %d negative samples" %( len( glob.glob1 ( train_pos_ph , "*.png" ) ), len( glob.glob1 ( train_neg_ph , "*.png" ) ) )
# (2c) 
# Preprocess and rescale test images, store them in test_ph, rescale column annotations, store them in test_gt_ph
# annotations_ph: Annotations of original shelf images with respect to MATLAB indexing must be in this path	
# 				  Image name and corresponding annotation file must have the same name
print "Creating test images with common height %d" %(boundary_shape[0])	
for idx, image_path in enumerate(test_listing):
	sys.stdout.write('Done: %d/%d\r'%( idx , len( test_listing ) ) )
	im = img_as_ubyte( io.imread ( os.path.join( images_ph , image_path ) , as_grey = True ) )
	image_name = os.path.splitext( image_path )[0]
	cols = np.ravel( sio.loadmat( os.path.join( annotations_ph , image_name + '.mat') )['cols'] )
	cols -= 1 # Conversion from MATLAB indexing to Python indexing
	( ymax , _ ) = im.shape
	rescale_ratio = float( boundary_shape[0] ) / float( ymax )
	cols = cols * rescale_ratio                            # rescale the cols as well
	np.save( os.path.join ( test_gt_ph , "test_" + str(idx) + ".npy") , cols)
	im = transform.rescale( im , rescale_ratio )            # Rescale the image
	im = filters.gaussian( im , sigma = 1 )                # Apply Gaussian filter
	im = exposure.equalize_hist( im )                      # Histogram equalization for illumination differences
	io.imsave( os.path.join(test_ph , "test_" + str( idx ) + ".png" ) , im )
	sys.stdout.flush()

print "Created test samples and rescaled annotations"

log = { "train_ratio" : train_ratio, "boundary_shape" : boundary_shape, 
"sampling_seed" : sampling_seed, "num_neg_sample_per_image" : num_neg_sample_per_image }


with open( dataset_log_ph , 'w' ) as outfile:
	json.dump( log , outfile )

print "Created dataset creation log"
tock = time.clock()
print "Elapsed Time:"
print tock - tick




