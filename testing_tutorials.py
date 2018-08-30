import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import os
from skimage import io as skio
import skimage
import skimage.segmentation
from skimage import data, filters, measure
import scipy
#plt.show()

def find_blobs(image):
    blobs = scipy.ndimage.find_objects(image)
    print(blobs)

def plot_squares(image):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in measure.regionprops(image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatch.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_blobs():
    n = 9
    l = 200
    im = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = filters.gaussian(im, sigma=l / (4. * n))
    blobs = im > im.mean()
    #all_labels = measure.label(blobs)
    blobs_labels, segments = measure.label(blobs, background=0, return_num=True)
    print("there were {} segments".format(segments))

    # remove artifacts connected to image border
    cleared_blobs = skimage.segmentation.clear_border(blobs_labels)
    plt.imshow(cleared_blobs, interpolation='nearest')
    plt.show()
    plot_squares(cleared_blobs)

def plot_mask():
    filename = os.path.join(skimage.data_dir, 'camera.png')
    camera = skimage.data.camera()
    val = filters.threshold_otsu(camera)
    mask = camera < val
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_save_disp():
    logo = skio.imread('http://scikit-image.org/_static/img/logo.png')
    skio.imsave('local_logo.png', logo)
    plt.imshow(logo, cmap='gray', interpolation='nearest')
    plt.show()

plot_blobs()