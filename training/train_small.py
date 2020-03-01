import sys
sys.path.append("/Path To /MobileNet-ssd-keras")
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from models.ssd_mobilenet import ssd_300
from models.small_ssd import small_ssd
from misc.keras_ssd_loss import SSDLoss, FocalLoss, weightedSSDLoss, weightedFocalLoss
from misc.keras_layer_AnchorBoxes import AnchorBoxes
from misc.keras_layer_L2Normalization import L2Normalization
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from misc.ssd_batch_generator import BatchGenerator
from keras.utils.training_utils import multi_gpu_model
import os
import keras
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

img_height = 300  # Height of the input images
img_width = 300 # Width of the input images
img_channels = 1  # Number of color channels of the input images
subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
#swap_channels = True  # The color channel order in the original SSD is BGR
n_classes = 1  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
#scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
#              1.05]
scales = [1.0, 1.0]

#scales = scales_voc

aspect_ratios = [ [1.0, 2.0, 0.5], 
				  [1.0, 2.0, 0.5]
				] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = False

#steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
#offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
#           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
#variances = [0.1, 0.1, 0.2,
#            0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

def train(args):
  model = small_ssd(mode = 'training',
  				  alpha = 0.25,
                  image_size=(img_height, img_width, img_channels),
                  n_classes=n_classes,
                  l2_regularization=0.0005,
                  scales=scales,
                  aspect_ratios_per_layer=aspect_ratios,
                  two_boxes_for_ar1=two_boxes_for_ar1,
                  steps=steps,
                  offsets=offsets,
                  limit_boxes=limit_boxes,
                  variances=variances,
                  coords=coords,
                  normalize_coords=normalize_coords,
                  subtract_mean=subtract_mean,
                  divide_by_stddev=None,
                  swap_channels=swap_channels)

  

  model.load_weights(args.weight_file, by_name=True,skip_mismatch=True)


  predictor_sizes = [model.get_layer('conv11_mbox_conf').output_shape[1:3],
                     model.get_layer('conv13_mbox_conf').output_shape[1:3],
                     model.get_layer('conv14_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv15_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv16_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv17_2_mbox_conf').output_shape[1:3]]
 
  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

  ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)


  model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


  train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
  val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

  # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

  # TODO: Set the paths to the datasets here.



