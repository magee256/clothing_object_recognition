from time import perf_counter
from utilities.io import Labels
from utilities.preproc import *

def preprocess_images(labels, chunksize):
    """
    Performs the steps necessary to preprocess images and 
    saves the processed images to disk. 

    Input:
    labels - Labels object with references to images and path info
    chunksize - The number of images to process at once

    Side Effect:
    Processed images stored to disk
    """
    ## Save processed images to disk as jpeg
    start = perf_counter(); image_count = 0
    labels.set_data_target('raw_image', chunksize)
    for i in range(labels.n_chunk):
        img_df = next(labels)
        img_df = preprocess_data(img_df)
        labels.save(img_df)
        
        image_count += len(img_df)
        print('Chunk {} of {} complete'.format(labels.i_chunk, labels.n_chunk))
    print('Took {} seconds to preprocess {} images'.format(
        perf_counter() - start,image_count))


if __name__ == '__main__':
    # Set up labels
    chunksize = 50
    n_images_loaded = 150  # -1 loads all
    image_to_category = 'data/Category and Attribute Prediction Benchmark' \
                        '/Anno/list_category_img.txt'
    image_path_prefix = 'data/Category and Attribute Prediction Benchmark' \
                        '/Img/Img/'
    labels = Labels(image_to_category, image_path_prefix, n_images_loaded)

    preprocess_images(labels, chunksize)
