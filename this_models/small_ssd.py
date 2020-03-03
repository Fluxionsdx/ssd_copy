import sys
sys.path.append("/Users/Josh/MobileNet-ssd-keras")
import keras
import numpy as np 
import cv2
import keras.backend as K
import keras.layers as KL
#from models.depthwise_conv2d import DepthwiseConvolution2D
from keras.models import Model
from keras.layers import Input, Lambda, Activation,Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,BatchNormalization, Add, Conv2DTranspose
from keras.regularizers import l2
from keras.applications import mobilenet
#from models.mobilenet_v1 import mobilenet
from misc.keras_layer_L2Normalization import L2Normalization
from misc.keras_layer_AnchorBoxes import AnchorBoxes
from misc.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast



def small_ssd(mode,
			alpha,
            image_size,
            n_classes,
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=False,
            steps=None,
            offsets=None,
            limit_boxes=False,
            coords='centroids',
            normalize_coords=False,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=False,
            return_predictor_sizes=False,
            variances=[1.0, 1.0, 1.0, 1.0]):


    n_predictor_layers = 2  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")


    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers


    feature_layer_1_name = "conv_pw_5_relu"
    feature_layer_2_name = "conv_pw_6_relu"
    mn = mobilenet.MobileNet(weights="imagenet", include_top=False, alpha=alpha, input_shape=image_size)
    feature_layer_1 = mn.get_layer(feature_layer_1_name).output
    feature_layer_2 = mn.get_layer(feature_layer_2_name).output

    # Feed conv4_3 into the L2 normalization layer
    # feature_layer_1 = L2Normalization(gamma_init=20, name='feature_layer_1')(feature_layer_1)
    feature_layer_1_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='feature_layer_1_mbox_conf')(feature_layer_1)
    feature_layer_2_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='feature_layer_2_mbox_conf')(feature_layer_2)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    feature_layer_1_mbox_loc = Conv2D(n_boxes[0] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='feature_layer_1_mbox_loc')(feature_layer_1)
    feature_layer_2_mbox_loc = Conv2D(n_boxes[1] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='feature_layer_2_mbox_loc')(feature_layer_2)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    feature_layer_1_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], limit_boxes=limit_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='feature_layer_1_mbox_priorbox')(feature_layer_1_mbox_loc)
    feature_layer_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    limit_boxes=limit_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='feature_layer_2_mbox_priorbox')(feature_layer_2_mbox_loc)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    feature_layer_1_mbox_conf_reshape = Reshape((-1, n_classes), name='feature_layer_1_mbox_conf_reshape')(
        feature_layer_1_mbox_conf)
    feature_layer_2_mbox_conf_reshape = Reshape((-1, n_classes), name='feature_layer_2_mbox_conf_reshape')(feature_layer_2_mbox_conf)

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    feature_layer_1_mbox_loc_reshape = Reshape((-1, 4), name='feature_layer_1_mbox_loc_reshape')(feature_layer_1_mbox_loc)
    feature_layer_2_mbox_loc_reshape = Reshape((-1, 4), name='feature_layer_2_mbox_loc_reshape')(feature_layer_2_mbox_loc)


    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    feature_layer_1_mbox_priorbox_reshape = Reshape((-1, 8), name='feature_layer_1_mbox_priorbox_reshape')(
        feature_layer_1_mbox_priorbox)
    feature_layer_2_mbox_priorbox_reshape = Reshape((-1, 8), name='feature_layer_2_mbox_priorbox_reshape')(feature_layer_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([feature_layer_1_mbox_conf_reshape,
                                                       feature_layer_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([feature_layer_1_mbox_loc_reshape,
                                                     feature_layer_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([feature_layer_1_mbox_priorbox_reshape,
                                                               feature_layer_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=mn.inputs, outputs=predictions)
    # return model

    if mode == 'inference':
        print ('in inference mode')
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=0.01,
                                                   iou_threshold=0.45,
                                                   top_k=100,
                                                   nms_max_output_size=100,
                                                   coords='centroids',
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        print ('in training mode')

    return model





















