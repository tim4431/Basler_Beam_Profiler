import cv2, numpy as np
from fit_gaussian import *


def blob_keypoint_detector(im):
    """
    Blob detection using OpenCV's SimpleBlobDetector.
    OpenCV's SimpleBlobDetector only supports CV_8U, so we create a normalized temporary version for detection.
    """
    if im.dtype == np.uint16:
        im_norm = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # print("Converted to uint8")
    else:
        im_norm = im

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100000000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.filterByColor = True
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)
    #
    keypoints = detector.detect(im_norm)
    return keypoints


def blur_threshold_morph(im, Ng=21, Nm=11):
    """
    Gaussian blur, threshold, and morphological operations
    """
    im = cv2.GaussianBlur(im, (Ng, Ng), 0)
    kernel = np.ones((Nm, Nm), np.uint8)
    # Apply threshold differently for 8-bit and 16-bit images
    if im.dtype == np.uint16:
        _, im = cv2.threshold(
            im, 1000, 65535, cv2.THRESH_BINARY
        )  # Adjust threshold based on intensity range
    else:
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    return im


def keypts_to_spots(keypoints):
    # Convert keypoints to a list of (x, y, radius)
    spots = []
    for kp in keypoints:
        x, y = kp.pt
        radius = kp.size / 2.0
        spot = (x, y, None, radius)  # in pixels
        spots.append(spot)
    return spots


def fit_gaussian_within_roi(img, spot, plot=False, ax=None):
    x, y, _, w = spot
    M = 2
    x0, x1, y0, y1 = int(x - M * w), int(x + M * w), int(y - M * w), int(y + M * w)
    # convert to CV_32F for cv2.getRectSubPix
    img = np.array(img, dtype=np.float32)
    img_crop = cv2.getRectSubPix(img, (x1 - x0, y1 - y0), (x, y))
    # coarse graining to 20*20 pixels
    Nc = 40
    Z = cv2.resize(img_crop, (Nc, Nc))
    X, Y = np.meshgrid(np.linspace(x0, x1, Nc), np.linspace(y0, y1, Nc))
    # plt.matshow(Z)
    # plt.show()

    mu, cov = statistics_for_gaussian2d(X, Y, Z)
    # eigval = np.linalg.eigvals(cov)
    eigval, eigvec = np.linalg.eig(cov)
    sigma_0, sigma_1 = np.sqrt(4 * eigval)
    vec_0, vec_1 = eigvec[:, 0], eigvec[:, 1]
    if sigma_0 < sigma_1:
        sigma_0, sigma_1 = sigma_1, sigma_0
        vec_0, vec_1 = vec_1, vec_0
    # sigma = np.sqrt(sigma_0 * sigma_1)
    x = mu[0]
    y = mu[1]
    spot = {
        "x": x,
        "y": y,
        "sigma_0": sigma_0,
        "sigma_1": sigma_1,
        "vec_0": vec_0,
        "vec_1": vec_1,
    }
    # print(spot)
    if plot:
        bounds_x = (np.min(X), np.max(X))
        bounds_y = (np.min(Y), np.max(Y))
        X_new = np.linspace(*bounds_x, 100)
        Y_new = np.linspace(*bounds_y, 100)
        X_new, Y_new = np.meshgrid(X_new, Y_new)
        Z_new = np.array(
            [
                gaussian_2d(x_, y_, mu, cov)
                for x_, y_ in zip(X_new.ravel(), Y_new.ravel())
            ]
        ).reshape(X_new.shape)
        #
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(
            Z / np.max(Z),
            origin="lower",
            extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]],
        )
        ax.contour(X_new, Y_new, Z_new, cmap="jet")

    return spot


def blob_detector(im, plot=False):
    im_blur = blur_threshold_morph(im)
    keypoints = blob_keypoint_detector(im_blur)
    spots = keypts_to_spots(keypoints)
    spots = [fit_gaussian_within_roi(im, spot, plot=plot) for spot in spots]
    return spots
