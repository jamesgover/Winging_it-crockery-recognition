import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import os
from skimage import io as skio
import skimage
import skimage.segmentation
from skimage import data, filters, measure
import scipy
import pandas as pd
from skimage.color import rgb2gray
#plt.show()


def find_blobs(image):
    blobs = scipy.ndimage.find_objects(image)
    print(blobs)


def plot(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


def create_pd_frame(region=False):
    if region is False:
        frame = pd.DataFrame(columns=["Area", "Orientation", "BBoxX", "BBoxY", "Type_o_Object"])
    else:
        minr, minc, maxr, maxc = region.bbox
        frame = pd.DataFrame([[region.area, region.orientation, maxr - minr, maxc - minc, False]],
                             columns=["Area", "Orientation", "BBoxX", "BBoxY", "Type_o_Object"])
    #print(frame)
    return frame


def plot_squares(image):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    data = create_pd_frame()
    #useful link == https://au.mathworks.com/help/images/ref/regionprops.html
    #print(measure.regionprops(image)[0])
    for region in measure.regionprops(image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatch.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            region_data = create_pd_frame(region=region)
            data = data.append(region_data, ignore_index=True)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return data

def make_blobs():
    n = 9
    l = 200
    im = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = filters.gaussian(im, sigma=l / (4. * n))
    blobs = im > im.mean()
    return blobs

def sep_and_strip_img(image):
    image_labels, segments = measure.label(image, background=0, return_num=True)
    print("there were {} segments".format(segments))

    # remove artifacts connected to image border
    cleared_labeled = skimage.segmentation.clear_border(image_labels)
    return cleared_labeled
    #plt.imshow(cleared_labeled, interpolation='nearest')
    #plt.show()
    #plot_squares(cleared_labeled)

def get_mask(image):
    image = rgb2gray(image)
    val = filters.threshold_otsu(image)
    mask = image < val
    return mask

def load_save_disp():
    logo = skio.imread('http://scikit-image.org/_static/img/logo.png')
    skio.imsave('local_logo.png', logo)
    plt.imshow(logo, cmap='gray', interpolation='nearest')
    plt.show()


#plot_squares(sep_and_strip_img(make_blobs()))
bowl_balls = skio.imread("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsuOCvBFxo8VQhrsJzcfjPHhy8ffPI0h3Mi__JXfytkwhHstVi")
data = plot_squares(sep_and_strip_img(get_mask(bowl_balls)))

plot(bowl_balls)