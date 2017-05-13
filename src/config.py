import ConfigParser as cp
import json

config = cp.RawConfigParser()
config.read('../data/config/config.cfg')

#  ------           [PATHS]          ------------- # 
pos_feat_ph_hog = config.get("paths", "pos_feat_ph_hog")
neg_feat_ph_hog = config.get("paths", "neg_feat_ph_hog")
pos_feat_ph_lbp = config.get("paths", "pos_feat_ph_lbp")
neg_feat_ph_lbp = config.get("paths", "neg_feat_ph_lbp")

model_path_logreg = config.get("paths", "model_path_logreg")
model_path_svm = config.get("paths", "model_path_svm")
model_path_ada = config.get("paths", "model_path_ada")
model_path = config.get("paths", "model_path")
pca_path_hog = config.get("paths", "pca_path_hog")
pca_path_lbp = config.get("paths", "pca_path_lbp")

annotations_ph = config.get("paths","annotations_ph")
images_ph = config.get("paths","images_ph")
train_pos_ph = config.get("paths","train_pos_ph")
train_neg_ph = config.get("paths","train_neg_ph")
test_ph = config.get("paths","test_ph")
test_gt_ph = config.get("paths","test_gt_ph")

dataset_log_ph = config.get("paths" , "dataset_log_ph")
lbp_log_ph = config.get("paths" , "lbp_log_ph")
hog_log_ph = config.get("paths" , "hog_log_ph")

#  ------           [HOG]          ------------- #
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
transform_sqrt = config.getboolean("hog", "transform_sqrt")

#  ------           [DATASET]          ------------- #
train_ratio = config.getfloat("dataset", "train_ratio")
boundary_shape = json.loads(config.get("dataset", "boundary_shape"))
sampling_seed = config.getint ("dataset", "sampling_seed")
num_neg_sample_per_image = config.getint("dataset", "num_neg_sample_per_image")

#  ------           [LBP]          ------------- #
num_pts = config.getint("lbp", "num_pts")
radius = config.getint("lbp", "radius")
method_lbp = config.get("lbp", "method_lbp")

#  ------           [PROCESS]          ------------- #
classifier_type = config.get("process", "classifier_type")
svm_kernel = config.get("process" , "svm_kernel")
feat_type = config.get("process", "feat_type")
pca_flag = config.getboolean("process", "pca_flag")
pca_dim_hog = config.getint("process", "pca_dim_hog")
pca_dim_lbp = config.getint("process", "pca_dim_lbp")
tp_tolerance = config.getint ("process", "tp_tolerance")
argrelmax_order = config.getint ("process", "argrelmax_order")
