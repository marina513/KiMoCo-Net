import numpy as np
from scipy.ndimage import sobel
from skimage.feature import canny
from scipy.ndimage import convolve
def crop_img(img):
    '''
    Parameters
    ----------
    img : numpy array
        Image to be cropped.
    
    Returns
    -------
    crop_img : numpy array
        Cropped image such that all slices 
        contain at least one non-zero entry
    '''
    #Indices where the img is non-zero
    indices = np.array(np.where(img>0))
    
    #Max and Min x values where img is non-zero
    xmin = np.min(indices[0])
    xmax = np.max(indices[0])
    #Max and Min y values where img is non-zero
    ymin = np.min(indices[1])
    ymax = np.max(indices[1])

    #Return cropped img
    return img[xmin:xmax, ymin:ymax]

def bin_img(img, n_levels = 128):
    '''
    Parameters
    ----------
    img : numpy array
        Image to bin.
    n_levels : int
        Number of levels to bin the intensities in
    
    Returns
    -------
    binned_img : numpy array
        Binned image, which has n_levels different 
        intensity values
    '''
    
    #Intensity values to map to
    vals, bins = np.histogram(img, bins = n_levels)

    #Bin image
    binned_img = bins[np.digitize(img, bins, right = True)]
    
    #Return binned image
    return binned_img

def calc_gradient_magnitude(img, mode="2d"):
    """Calculate the magnitude of the image gradient.
    Note:
        - The image is assumed to be masked and normalised to [0, 1].
        - The image is converted to floating point numbers for a correct
        calculation of the gradient.
    """

    img = img.astype(float)

    grad_x = sobel(img, axis=0, mode='reflect')
    grad_y = sobel(img, axis=1, mode='reflect')

    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def tenengrad(img, brainmask=None, reduction='mean'):
    """Tenengrad measure of the input image.

    The code is based on the article:
    Krotkov E. Focusing. Int J Comput Vis. 1988; 1(3):223-237

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : boolean True or False, optional
        If True, a brainmask was used to mask the images before 
        calculating the metrics. Image is flattened prior metric 
        estimation. The default is False.

    Returns
    -------
    tg : float: Tenengrad measure of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    if brainmask is not None:
        grad = np.ma.masked_array(grad, mask=(brainmask != 1))

    return np.mean(grad ** 2),grad
















def gradient_entropy(img, brainmask=None, reduction='mean'):
    """
    Calculate gradient entropy of an image.

    Reference:
    Atkinson, D.; Hill, D. L.; Stoyle, P. N.; Summers, P. E.; Keevil, S. F.
    (1997): Automatic correction of motion artifacts in magnetic resonance
    images using an entropy focus criterion. In IEEE Transactions on Medical
    Imaging 16 (6), pp. 903–910. DOI: 10.1109/42.650886.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : bool, optional
        Whether the metric values should be masked with the brainmask. If None,
        no masking is performed.
    reduction : str, optional
        Reduction method for the image entropy values of multiple slices.
        Options: 'mean' (default), 'worst'.

    Returns
    -------
    ie : float
        Image Entropy of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    ge_slices = []
    for sl in range(img.shape[0]):
        if brainmask is not None:
            grad_slice = grad[sl][brainmask[sl] == 1]
        else:
            grad_slice = grad[sl].flatten()

        norm_intensity = grad_slice / np.sqrt(np.sum(grad_slice ** 2))
        ge_slices.append(-np.nansum(norm_intensity * np.log(norm_intensity)))

    if reduction == 'mean':
        return np.mean(ge_slices)
    elif reduction == 'worst':
        return np.max(ge_slices)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def normalized_gradient_squared(img, brainmask=None, reduction='mean'):
    """Normalized gradient squared measure of the input image.

    The code is based on the article:
    McGee K, Manduca A, Felmlee J et al. Image metric-based correction
    (autocorrection) of motion effects: analysis of image metrics. J Magn Reson
    Imaging. 2000; 11(2):174-181.

    Note:
    - The value of the metric is scaled by the number of pixels in the image
       to avoid the value being too small.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : bool, optional
        If True, a brainmask was used to mask the images before
        calculating the metrics. Image is flattened prior metric
        estimation. The default is False.

    Returns
    -------
    ngs : float
        Normalized gradient squared measure of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    if brainmask is not None:
        grad = np.ma.masked_array(grad, mask=(brainmask != 1))

    ngs_values = np.sum((grad / np.sum(grad)) ** 2, axis=(1, 2)) * grad[0].size

    if reduction == 'mean':
        return np.mean(ngs_values)
    elif reduction == 'worst':
        return np.min(ngs_values)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def aes(img, brainmask=None, sigma=np.sqrt(2), n_levels=128, bin=False,
        crop=True, weigt_avg=False, reduction='mean'):
    """
    Calculate the metric Average Edge Strength.
    Original code my Simon Chemnitz-Thomsen.

    Reference:
    Quantitative framework for prospective motion correction evaluation
    Nicolas Pannetier, Theano Stavrinos, Peter Ng, Michael Herbst,
    Maxim Zaitsev, Karl Young, Gerald Matson, and Norbert Schuff

    Parameters
    ----------
    img : numpy array
        Image for which the metrics should be calculated.
    brainmask : numpy array
        Brainmask for the image. If provided, the metric will be calculated
        only on the masked region.
    sigma : float
        Standard deviation of the Gaussian filter used
        during canny edge detection.
    n_levels : int
        Levels of intensities to bin image by
    bin : bool
        Whether to bin the image
    crop : bool
        Whether to crop image/ delete empty slices
    weigt_avg : bool
        Whether to calculate the weighted average (depending on the
        proportion of non-zero pixels in the slice).
    reduction : str
        Method to reduce the edge strength values.
        'mean' or 'worst'

    Returns
    -------
    AES : float
        Average Edge Strength measure of the input image.
    """
    img = np.array(img.squeeze().cpu().detach())

    # Crop image if crop is True (and no brainmask provided)
    if crop:
        if brainmask is None:
            img = crop_img(img)

    # Bin image if bin is True
    if bin:
        img = bin_img(img, n_levels=n_levels)

    # Centered Gradient kernel in the y- and x-direction
    y_kern = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    x_kern = y_kern.T


    # weights for each slice (proportion of non zero pixels)
    weights = []

    img = img.astype(float)

    # Weight, proportion of non zero pixels
    weights.append(np.mean(img > 0))

    # Convolve slice
    x_conv = convolve(img, x_kern)
    y_conv = convolve(img, y_kern)

    # Canny edge img
    canny_img = canny(img, sigma=sigma)

    if brainmask is not None:
        canny_img = np.ma.masked_array(canny_img,
                                        mask=(brainmask[:] != 1))
        x_conv = np.ma.masked_array(x_conv,
                                    mask=(brainmask != 1))
        y_conv = np.ma.masked_array(y_conv,
                                    mask=(brainmask != 1))

    # Numerator and denominator, to be divided
    # defining the edge strength of the slice
    grad= (x_conv ** 2 + y_conv ** 2) # more  effect on error suprressing
    grad= calc_gradient_magnitude(img)

    numerator = np.sum(canny_img * grad)
    denominator = np.sum(canny_img)

    es = np.array( np.sqrt(numerator) / denominator )
    # Remove nans
    es = es[~np.isnan(es)]

    if weigt_avg:
        return np.average(es, weights=weights)
    else:
        return np.mean(es), canny_img, grad
