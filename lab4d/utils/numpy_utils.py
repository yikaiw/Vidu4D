# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np


def interp_wt(x, y, x2, type="linear"):
    """Map a scalar value from range [x0, x1] to [y0, y1] using interpolation

    Args:
        x: Input range [x0, x1]
        y: Output range [y0, y1]
        x2 (float): Scalar value in range [x0, x1]
        type (str): Interpolation type ("linear" or "log")
    Returns:
        y2 (float): Scalar value mapped to [y0, y1]
    """
    # Extract values from tuples
    x0, x1 = x
    y0, y1 = y

    # # Check if x2 is in range
    # if x2 < x0 or x2 > x1:
    #     raise ValueError("x2 must be in the range [x0, x1]")

    if type == "linear":
        # Perform linear interpolation
        y2 = y0 + (x2 - x0) * (y1 - y0) / (x1 - x0)

    elif type == "log":
        # Transform to log space
        log_y0 = np.log10(y0)
        log_y1 = np.log10(y1)

        # Perform linear interpolation in log space
        log_y2 = log_y0 + (x2 - x0) * (log_y1 - log_y0) / (x1 - x0)

        # Transform back to original space
        y2 = 10**log_y2

    else:
        raise ValueError("interpolation_type must be 'linear' or 'log'")

    y2 = np.clip(y2, np.min(y), np.max(y))
    return y2


def pca_numpy(raw_data, n_components):
    """Return a function that applies PCA to input data, based on the principal
    components of a raw data distribution.

    Args:
        raw_data (np.array): Raw data distribution, used to compute
            principal components.
        n_components (int): Number of principal components to use
    Returns:
        apply_pca_fn (Function): A function that applies PCA to input data
    """
    # center the data matrix by subtracting the mean of each feature
    mean = np.mean(raw_data, axis=0)
    centered_data = raw_data - mean

    # compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data.T)

    # compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # sort the eigenvalues in descending order and sort the eigenvectors accordingly
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # choose the top k eigenvectors (or all eigenvectors if k is not specified)
    top_eigenvectors = sorted_eigenvectors[:, :n_components]

    def apply_pca_fn(data, normalize=False):
        """
        Args:
            data (np.array): Data to apply PCA to
            normalize (bool): If True, normalize the data to 0,1 for visualization
        """
        shape = data.shape
        data = data.reshape(-1, shape[-1])
        data = np.dot(data - mean, top_eigenvectors)

        if normalize:
            # scale to std = 1
            data = data / np.sqrt(eigenvalues[sorted_indices][:n_components])
            data = np.clip(data, -2, 2)  # clip to [-2, 2], 95.4% percentile
            # scale to 0,1
            data = (data + 2) / 4

        data = data.reshape(shape[:-1] + (n_components,))
        return data

    return apply_pca_fn


def bilinear_interp(feat, xy_loc):
    """Sample from a 2D feature map using bilinear interpolation

    Args:
        feat: (H,W,x) Input feature map
        xy_loc: (N,2) Coordinates to sample, float
    Returns:
        feat_samp: (N,x) Sampled features
    """
    dtype = feat.dtype
    ul_loc = np.floor(xy_loc).astype(int)  # x,y
    x = (xy_loc[:, 0] - ul_loc[:, 0])[:, None]  # (N, 1)
    y = (xy_loc[:, 1] - ul_loc[:, 1])[:, None]  # (N, 1)
    ul_loc = np.clip(ul_loc, 0, 110)  # clip
    q11 = feat[ul_loc[:, 1], ul_loc[:, 0]]  # (N, 16)
    q12 = feat[ul_loc[:, 1], ul_loc[:, 0] + 1]
    q21 = feat[ul_loc[:, 1] + 1, ul_loc[:, 0]]
    q22 = feat[ul_loc[:, 1] + 1, ul_loc[:, 0] + 1]
    feat_samp = (
        q11 * (1 - x) * (1 - y)
        + q21 * (1 - x) * (y - 0)
        + q12 * (x - 0) * (1 - y)
        + q22 * (x - 0) * (y - 0)
    )
    feat_samp = feat_samp.astype(dtype)
    return feat_samp
