from skimage.transform import rescale

def preprocess_data(img_df):
    img_df['data'] = scale_images(img_df['data'])
    img_df['data'] = fill_image(img_df['data'])
    return img_df

def scale_images(img_series):
    """
    Scale images so first dimension is 224 for input to ImageNet
    """
    def first_dim_224(img):
        """Scale first dimension of image to 224 pixels"""
        height = img.shape[0]
        return rescale(img,224/height,mode='constant')
    rescaled = img_series.apply(first_dim_224)
    return rescaled

def pad_images(img_series):
    """
    Pad second dimension of images with black until length 224
    """
    def pad_with_black(img):
        """
        If image's x dim is less than 224 pixels pad with black. 
        Modification spread evenly across both sides.
        """
        pix_to_pad = 224 - img.shape[1]
        if pix_to_pad > 0:
            img = np.pad(img,
                         [(0,0), ((pix_to_pad+1)//2, pix_to_pad//2), (0,0)],
                         mode='constant')
        return img
    padded = img_series.apply(pad_with_black)
    return padded

def crop_images(img_series):
    """Crop second dimension of images until length 224 reached"""
    def crop_image(img):
        """
        Crop image's x dim to 224 pixels. 
        Modification spread evenly across both sides.
        """
        pix_to_pad = 224 - img.shape[1]
        if pix_to_pad < 0:
            pix_to_crop = -pix_to_pad
            img = img[:,pix_to_crop//2:-((pix_to_crop+1)//2),:]
        return img
    cropped = img_series.apply(crop_image)
    return cropped
