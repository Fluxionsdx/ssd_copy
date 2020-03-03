import sys
#sys.path.append("/Users/Josh/Mobilenet-ssd-keras")
sys.path.append("/projects/mines/Josh/ssd_copy")
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

#from models.ssd_mobilenet import ssd_300
from this_models.small_ssd import small_ssd
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
img_channels = 3  # Number of color channels of the input images
subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
swap_channels = False  # The color channel order in the original SSD is BGR
n_classes = 1  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
#scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
#              1.05]
scales = [1.0, 1.0, 2.0]

#scales = scales_voc

aspect_ratios = [ [1.0, 2.0, 0.5], 
				  [1.0, 2.0, 0.5]
				] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = False

#steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
#offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
#           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0,
           1.0]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True
data_type = "sonar"
sonar_range = "range5000"
datasets_train = [1,3,4,5,6,7,11,12]
datasets_val = [2]

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
                  two_boxes_for_ar1=False,
                  steps=None,
                  offsets=None,
                  limit_boxes=limit_boxes,
                  variances=variances,
                  coords=coords,
                  normalize_coords=normalize_coords,
                  subtract_mean=subtract_mean,
                  divide_by_stddev=None,
                  swap_channels=swap_channels)

  


  predictor_sizes = [model.get_layer('feature_layer_1_mbox_conf').output_shape[1:3],
                     model.get_layer('feature_layer_2_mbox_conf').output_shape[1:3],]
 
  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

  ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)


  model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


  train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
  val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

  # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

  # TODO: Set the paths to the datasets here.

  #Create list of all image dirs
  train_images_dirs = []
  train_annotations_dirs = []
  train_filenames = []
  val_images_dirs = []
  val_annotations_dirs = []  
  val_filenames = []
  if(data_type == "sonar"):
  	if(sonar_range == "range5000"):
  		base_im_path = "/projects/mines/working_mount/processed_sonar/new_data"
  		base_an_path = "/projects/mines/Josh/mines_ground_truth/sonar/range5000"
  		base_filenames_path ="/projects/mines/Josh/mines_file_names"
		#base_im_path = "/Users/Josh/processed_sonar"
  		#base_an_path = "/Users/Josh/mines_ground_truth/sonar/range5000"
  		#base_filenames_path = "/Users/Josh/mines_file_names"
  		all_datasets = datasets_train + datasets_val
		for ds in all_datasets:
			#im_ds_path = "{}/k-8".format(ds)
			im_ds_path = "{}/range5000/k-8".format(ds)
			im_path = "{}/{}".format(base_im_path, im_ds_path)
			an_path = "{}/{}".format(base_an_path, ds)
			file_name_path = "{}/sonar_ds_{}_list.txt".format(base_filenames_path, ds)
			if ds in datasets_train:
  				train_images_dirs.append(im_path)  				
  				train_annotations_dirs.append(an_path)
  				train_filenames.append(file_name_path)
  			else:
  				val_images_dirs.append(im_path)  				
  				val_annotations_dirs.append(an_path)
  				val_filenames.append(file_name_path)


  # The XML parser needs to now what object class names to look for and in which order to map them to integers.

  classes = ['background',
             'mine']


  train_dataset.parse_xml(images_dirs=train_images_dirs,
                          image_set_filenames=train_filenames,
                          annotations_dirs=train_annotations_dirs,
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False,
                          data_type="sonar")


  val_dataset.parse_xml(images_dirs=val_images_dirs,
                        image_set_filenames=val_filenames,
                        annotations_dirs=val_annotations_dirs,
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        data_type="sonar")

  # 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

  ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                  img_width=img_width,
                                  n_classes=n_classes,
                                  predictor_sizes=predictor_sizes,
                                  min_scale=None,
                                  max_scale=None,
                                  scales=scales,
                                  aspect_ratios_global=None,
                                  aspect_ratios_per_layer=aspect_ratios,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  steps=None,
                                  offsets=None,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  pos_iou_threshold=0.5,
                                  neg_iou_threshold=0.2,
                                  coords=coords,
                                  normalize_coords=normalize_coords)

  batch_size = args.batch_size

  train_generator = train_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=False,
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(img_height, img_width, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

  val_generator = val_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=False,
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(img_height, img_width, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(img_height, img_width, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

  # Get the number of samples in the training and validations datasets to compute the epoch lengths below.
  n_train_samples = train_dataset.get_n_samples()
  n_val_samples = val_dataset.get_n_samples()



  def lr_schedule(epoch):
      if epoch <= 300:
          return 0.001
      else:
          return 0.0001


  learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
   
  checkpoint_path = args.checkpoint_path + "/ssd300_epoch-{epoch:02d}.h5"

  checkpoint = ModelCheckpoint(checkpoint_path)
  
  log_path = args.checkpoint_path + "/logs"

  tensorborad = TensorBoard(log_dir=log_path,
                            histogram_freq=0, write_graph=True, write_images=False)



  callbacks = [checkpoint,tensorborad,learning_rate_scheduler]

  # TODO: Set the number of epochs to train for.
  epochs = args.epochs
  intial_epoch = args.intial_epoch

  history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=ceil(n_train_samples)/batch_size,
                                verbose=1,
                                initial_epoch=intial_epoch,
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_steps=ceil(n_val_samples)/batch_size,
                                callbacks=callbacks
                                )

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--epochs',type=int,
                        help='Number of epochs', default = 500)
    parser.add_argument('--intial_epoch',type=int,
                        help='intial_epoch', default=0)
    parser.add_argument('--checkpoint_path',type=str,
                        help='Path to save checkpoint', default="./checkpoint")
    parser.add_argument('--batch_size',type=int,
                        help='batch_size', default=32)

    args = parser.parse_args()
    train(args)

