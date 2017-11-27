## Functions used in optimizing model hyperparameters
import os
from time import perf_counter
from utilities.io import Labels
from build_model import *

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import InputLayer
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

def stratified_train_val_test(y):
    """
    Generate stratified train-val-test splits. 
    Hard coded for 70-20-10 proportion
    
    Input:
    y - label array that train-val-test indices will be produced for

    Output:
    train - training set indices
    valid - validation set indices
    test - test set indices
    """
    # Train-test split
    sss = StratifiedShuffleSplit(n_splits=1,test_size=.1)
    train, test = next(sss.split(y,y))
    
    # Train-validate split
    sss = StratifiedShuffleSplit(n_splits=1,test_size=2/9)
    train, valid = next(sss.split(y[train],y[train]))
    return train, valid, test

def subset_labels(indices,labels,chunksize,name):
    """
    Subset a labels object and prepare it for reading bottleneck features.

    Create a Labels object from the input "labels" containing only
    the entries specified by "indices". The new Labels object's target is 
    then set for bottleneck features and set to return numpy arrays.

    Input:
    indices - The indices for the data subset
    labels - The Labels object that will be subsetted
    chunksize - The chunksize to use when reading values from the subset
    name - The model name

    Output:
    sub_labels - Subsetted Labels object
    """
    sub_labels = Labels(labels.labels.iloc[indices,:],
                        labels.image_path_prefix, labels.n_images_loaded)
    sub_labels.set_data_target('bottleneck',chunksize,name)
    sub_labels.iter_df = False
    return sub_labels

def optimize_model(model, name, train_labels, valid_labels,
                   max_epochs=5, pretrain_dense=False, load_trained=True):
    """
    Optimize the supplied model with layers unfrozen at a variety of depths.

    The best model is tracked over epochs and the best one for both the 
    output layer unfrozen and for any number of internal layers unfrozen
    is saved.

    Input:
    model - The keras Model to train
    name - The string to use for the model name
    train_labels - Labels object containing training data
    valid_labels - Labels object containing validation data
    max_epochs - The number of epochs to train each unfrozen non-output layer
    pretrain_dense - Train the output layer for extra iterations?
    load_trained - Use the previously saved weights as a starting point?

    Output:
    history_list - Training session history object

    Side Effect:
    Best models written to disk
    """
    # Output layer gets tuned first
    refit_modelf = 'intermediates/saved_models/test_{}_tuned_out_layer.hdf5'.format(name)
    tuned_modelf = 'intermediates/saved_models/test_{}_base_weights.hdf5'.format(name)
    
    if load_trained and os.path.exists(refit_modelf):
        # Load pre-trained output layer weights
        model.load_weights(refit_modelf)

    history_list = []
    if pretrain_dense or not os.path.exists(refit_modelf):
        checkpointer = ModelCheckpoint(
                filepath=refit_modelf, 
                verbose=1, save_best_only=True)
        unfreeze_model(model,1)
        
        history = model.fit_generator(train_labels,
                    steps_per_epoch=train_labels.n_chunk,
                    epochs=100,
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=valid_labels,
                    validation_steps=valid_labels.n_chunk,
                    )
        history_list.append(history)
        history_list.append(None)  # Marks end of pretraining history

    # Use best weights found so far
    model.load_weights(refit_modelf)
    
    checkpointer = ModelCheckpoint(
                filepath=tuned_modelf, 
                verbose=1, save_best_only=True)
    max_unfrozen = count_max_frozen(model)
    for i_unfrozen in range(2,max_unfrozen+1):
        unfreeze_model(model,i_unfrozen)
        
        history = model.fit_generator(train_labels,
                    steps_per_epoch=train_labels.n_chunk,
                    epochs=max_epochs,
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=valid_labels,
                    validation_steps=valid_labels.n_chunk,
                    )
        history_list.append(history)
        print('Model tuned with {} of {} layers unfrozen'
              .format(i_unfrozen,max_unfrozen))
    return history_list

def count_max_frozen(model):
    """
    Counts the total number of layers in model that have weights
    
    Input:
    model - The model we want to count the layers of

    Output:
    freeze_count - The number of layers available to freeze
    """
    freeze_count = 0
    for layer in model.layers:
        if layer.weights:
            freeze_count += 1
    return freeze_count

def unfreeze_model(model,n_unfrozen):
    """
    Unfreeze from the output layer to n_unfrozen layers in.

    Done according to the order specified in model.layers
    only counting layers with weights. 

    Input:
    model - The model that will have its layers unfrozen
    n_unfrozen - The number of layers to unfreeze

    Side Effect:
    n_unfrozen layers set to trainable
    """
    i_unfrozen = 0
    for layer in model.layers[::-1]:
        if layer.weights:
            layer.trainable = True
            i_unfrozen += 1
            if i_unfrozen == n_unfrozen:
                break
        
def prep_model(model,max_layer,depth):
    """
    Copy the last few layers of model and prepare the 
    resulting model for training.

    Input:
    model - The model used as reference
    max_layer - Layer with this name has its output replaced with input layer
    depth - The number of model outputs

    Output:
    model_top - The top of model, capable of taking input and with a new
                output layer
    """
    model_top = model_top_at_layer(model,max_layer)
    model_top = replace_out_layer(model_top,depth)
    for layer in model_top.layers:
        layer.trainable = False
    model_top.layers[-1].trainable = True
    model_top.compile('adam','categorical_crossentropy',metrics=['accuracy'])
    return model_top

def expand_labels(labels):
    """
    Prepare labels to read flipped bottleneck features and train.

    Input:
    labels - Labels object with only base image paths

    Output:
    labels - Labels object with flipped image paths added and OHE targets
    """
    # Append paths to flipped image bottleneck features
    flip_labels = labels.labels.copy()
    flip_labels['image_name'] = flip_labels['image_name'].apply(
                                lambda x: '_flip' + x)
    flip_labels = Labels(flip_labels,image_path_prefix,labels.n_images_loaded)
    labels = labels + flip_labels
    
    labels.one_hot_labels()
    return labels

def train_model(labels, chunksize):
    """
    Train the models specified by models_to_run on the images 
    supplied in labels. Model weights saved to disk. 

    Input:
    labels - Labels object containing references to data to train on
    chunksize - The number of images to train on at a time

    Output:
    hist_dict - Maps model name to its training history
    """
    labels = expand_labels(labels)
    hist_dict = {}
    train, valid, test = stratified_train_val_test(
                         np.stack(labels.labels['category_label'].values))
    
    max_layer_dict = {
        'resnet' : 'activation_46',
        'inception_v3' : 'mixed9',
        'xception' : 'add_28',
        }

    model_dict = {
        'resnet' : ResNet50(weights='imagenet'),
        'inception_v3' : InceptionV3(weights='imagenet'),
        'xception' : Xception(weights='imagenet'),
        }

    models_to_run = [
    #    'resnet', 
    #    'inception_v3',
        'xception',
    ]
    for name in models_to_run:
        print('Fine tuning {} model'.format(name))
        model = model_dict[name]
        max_layer = max_layer_dict[name]
        model_top = prep_model(model,max_layer,depth=10)
        
        train_labels = subset_labels(train,labels,chunksize,name)
        valid_labels = subset_labels(valid,labels,chunksize,name)
    
        start = perf_counter()
        hist_dict[name] = optimize_model(model_top,name,train_labels,
                          valid_labels,
                          pretrain_dense=True, load_trained=True)
        print('Took {} seconds to optimize all layers for {}'
              .format(perf_counter() - start,name))
    return hist_dict


if __name__ == '__main__':
    ## Train the top of the model on the selected labels
    np.random.seed(313) # For reproducibility
    
    chunksize = 3000
    n_images_loaded = 150  # -1 loads all
    image_to_category = 'data/Category and Attribute Prediction Benchmark' \
                        '/Anno/list_category_img.txt'
    image_path_prefix = 'data/Category and Attribute Prediction Benchmark' \
                        '/Img/Img/'
    labels = Labels(image_to_category, image_path_prefix, n_images_loaded)

    hist_dict = train_model(labels, chunksize)
