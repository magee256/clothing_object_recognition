import numpy as np
from functools import partial
from time import perf_counter
from utilities.io import Labels
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.applications.resnet50 import preprocess_input as res_preproc
from keras.applications.inception_v3 import preprocess_input as inc_preproc
from keras.applications.xception import preprocess_input as xcept_preproc

def create_bottleneck_model(model,model_name):
    """
    Modifies a model to output at an intermediate layer.

    Input:
    model - keras model bottleneck features desired from
    model_name - Name of model. Used to find max_layer name

    Output:
    Model set to output at an appropriate hidden layer
    """
    # Output layers chosen to be near output layer and at a 
    # point where tensor flows converge
    # WARNING: Fragile. Must load models only once in order above
    max_layer_dict = {
        'resnet' : 'activation_46',
        'inception_v3' : 'mixed9',
        'xception' : 'add_28',
        }

    output_layer = model.get_layer(max_layer_dict[model_name])
    return Model(model.input,output_layer.output)

def flip_img(img):
    """Rotates image across vertical axis"""
    return np.flip(img,axis=1)

def calc_bottleneck_features(labels,model_name,bottleneck_model,flip=False):
    """
    Calculate bottleneck features in chunks.

    Does this by predicting on images specified in labels using 
    bottleneck_model.
    
    Input:
    labels - Labels object containing image data
    model_name - The name of the model used as a base
    bottleneck_model - Truncated keras model
    flip - Whether to predict on flipped images or not

    Output:
    img_df - Dataframe with predictions for a chunk of data
    """
    for _ in range(labels.n_chunk):
        img_df = next(labels)
        img_df['data'] = img_df['data'].apply(partial(normalize_image_vals,model_name))
        if flip:
            img_df['data'] = img_df['data'].apply(flip_img)
            img_df['image_name'] = img_df['image_name'].apply(lambda x: '_flip' + x)
        bot_feats = bottleneck_model.predict(np.stack(img_df['data'].values))
        img_df['data'] = [ a.astype(np.float16) for a in bot_feats ]
        yield img_df
        
def normalize_image_vals(name,image):
    """Preprocess loaded images for feeding into pretrained imagenet models"""
    preproc_dict = {
        'resnet' : res_preproc,
        'inception_v3' : inc_preproc,
        'xception' : xcept_preproc,
    }
    image = preproc_dict[name](image.astype(np.float64))
    return image

def store_bottle_feats(labels, chunksize):
    """
    Calculate and store bottleneck features based off settings in labels.

    Truncates one of several Imagenet trained models then uses them
    to predict on the images referenced in labels. These predictions
    are stored to disk as .npy files using 16 bit precision floats. 

    Input:
    labels - Labels object with references to images and path info
    chunksize - The number of images to process at once

    Output:
    The bottleneck features stored to disk
    """
    model_dict = {
            'resnet' : ResNet50,
            'inception_v3' : InceptionV3,
            'xception' : Xception,
            }

    # Output from below roughly 90 GB, may need to run one at a time 
    # and transfer to different storage
    #models_to_run = [ 'resnet', 'inception_v3'] # Took 1449.48 s and 1179.66 s
    #models_to_run = ['xception'] # Took 1546.35 s
    models_to_run = ['resnet', 'inception_v3', 'xception']
    for name in models_to_run:
        model = model_dict[name](weights='imagenet')
        bottleneck_model = create_bottleneck_model(model, name)
        
        start = perf_counter()
        # For normal images
        labels.set_data_target('proc_image', chunksize, name)
        for bot_feat_df in calc_bottleneck_features(labels, name, bottleneck_model):
            labels.save(bot_feat_df)
            print('Chunk {} of {} complete'.format(labels.i_chunk, labels.n_chunk))
            
        # For flipped images
        labels.set_data_target('proc_image', chunksize, name)
        for bot_feat_df in calc_bottleneck_features(labels, name,
                           bottleneck_model, flip=True):
            labels.save(bot_feat_df)
            print('Chunk {} of {} complete'.format(labels.i_chunk,labels.n_chunk))
            
        print('Took {} seconds to calculate {} bottleneck features for {}'
              .format(perf_counter() - start,len(labels),name))


if __name__ == '__main__':
    # Set up labels
    chunksize = 5000
    n_images_loaded = 150  # -1 loads all
    image_to_category = 'data/Category and Attribute Prediction Benchmark' \
                        '/Anno/list_category_img.txt'
    image_path_prefix = 'data/Category and Attribute Prediction Benchmark' \
                        '/Img/Img/'
    labels = Labels(image_to_category, image_path_prefix, n_images_loaded)

    store_bottle_feats(labels, chunksize)
