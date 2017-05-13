## **Boundary Detection of Shelf Products**

This repository is created and developed for the term project of the EE475 "Introduction to Image Processing" course offered by Prof. Bulent Sankur of Electrical & Electronics Engineering Department of Bogazici University in the 2016-2017 Fall Term.

The project is undertaken in collaboration with Vispera Information Technologies ([website](http://vispera.co)). We would like to thank our supervisors at Vispera for offering this problem, providing us the data and their guidance throughout the project.

In this project, split-lines between products on shop racks images are detected using various types of feature extraction and data mining methods. The problem can be generalized to any type of object detection task that aims to pinpoint the width-location of any object standing on a horizontal surface like a shelf or a shop-rack (e.g. books in a library).

### *DATASET*

A typical dataset consists of JPG images cropped shelf by shelf and stored in *../data/images*. Positive annotations per image are stored in *../data/column_annotations* as a .MAT file with the same file name as the respective image.
Use *../src/dataset.py* to create positive and negative training samples as well as test images with rescaled annotations to get started with the project.  

### *MODULES*

To implement the methodology Python 2.7 is used. Code is tested on both OS X El Capitan and Windows 10 operating systems. Below you can see the necessary modules and their versions that were used in generating and testing this code:

- scikit-learn==0.17.1
- numpy==1.11.1
- configparser==3.5.0
- scikit-image==0.12.3
- scipy==0.17.1
- matplotlib==1.5.1

It is advised that you use a virtual environment to import aforementioned modules with respective versions in order to run the scripts seamlessly.

### *CONFIGURATIONS*

You can play with many options and parameters for your liking. To change these parameters you can use *../data/config.cfg* file. Some suggestions for parameters to play with are:
- **Boundary Shape**: Fix a height and play with width to observe its effect on precision and recall rates.
- **Sampling Seed**: Play with the seed to run analysis with different train and test samples
- **Number of Negative Sample per Image**: You may wish to have more number of negative samples in order to have a more precise classifier. Yet this increases your computation time substantially.
- **Classifier Type**: Different classifier algorithms may outperform others in certain situations.
- **LBP and HOG Feature Paremeters**: Play with them to assess the performance of features
- **Peak Localization Order**: You may wish to use the 'o' option of Test Stage to decide on this.
- **PCA Parameters:** You may wish to check the PCA Analysis stage to decide on dimensionality reduction parameters.

### *ALGORITHM STAGES*
Run the *../src/main.py* script to initiate the project after creating the train and test samples.

#### *Feature Extraction*
At this stage, features are extracted from train samples. If you haven't changed your dataset and feature extraction related configurations, you may wish to skip this stage. The algorithm will prompt you this option if that is the case. Feature options:
- **LBP:** Local Binary Patterns
- **HOG:** Histogram of Oriented Gradients
- **FUS:** Fusion. Formed by concatenating LBPs and HOGs *after* Principal Component Analysis (if enabled). 

#### *Principal Component Analysis*
At this stage, you may wish to investigate the principal component characteristics of your featureset. A scree plot is presented to you for guidance.
When you decide on to how many dimensions to reduce your datasets, check the *pca_dim_lbp* and *pca_dim_hog* parameters in *../data/config/config.cfg*. 

#### *Classification*
At this stage, classification training is done with the previously extracted features. Options for classification algorithms are:
- Logistic Regression ( config key : LOG_REG )
- Support Vector Machines with kernel options 'linear' , 'sigmoid' , 'rbf' , 'poly' ( config key : SVM + kernel_keys: 'rbf' or 'linear' or 'poly' or 'sigmoid')
- AdaBoost with Decisions Stumps as base estimator ( config key : ADA_BST )

#### *Testing*
At this stage you have three options:
- **M - Manual Mode:** Manually go through test images to inspect the predictions and signal visually
- **C - Calculation Mode:** Calculate precision , recall and f-measure metrics for the whole test set.
- **O - Optimum Peak Localization Order Mode:** For the given configuration, finds the optimum peak localization order and plots the results.

### *AUTHORS*

- Alper Bayram <alperbayram2@gmail.com>
- Dorukhan Sergin <dorukhan.sergin@gmail.com>
