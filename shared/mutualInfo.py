import numpy as np
def mutual_information(Img1,Img2):
    """ Mutual information for joint histogram
    """
    t1_slice = Img1.squeeze().cpu().detach()
    t2_slice = Img2.squeeze().cpu().detach()
    hgram, x_edges, y_edges = np.histogram2d(
        t1_slice.ravel(),
        t2_slice.ravel(),
        bins=20)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
