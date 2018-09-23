import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import os
from skimage import io as skio
import skimage
import skimage.segmentation
from skimage import filters, measure
import scipy
import pandas as pd
from skimage.color import rgb2gray
# plt.show()
TRAIN = True
MASK_SENSITIVITY = .2
MIN_AREA = 100000


'''simple function to let functions toggle the train variable
'''
def set_train(train):
    TRAIN = train


def get_item_string(item):
    if item is 1:
        item = 'bowl'
    elif item is 2:
        item = 'plate'
    elif item is 3:
        item = 'small_plate'
    elif item is 4:
        item = 'bread_bowl'
    elif item is 5:
        item = 'cup'
    else:
        item = "ERROR"
    return item


def label_obj(region, original, train=TRAIN):
    if train:
        # plot(region.filled_image())
        #plot(region.filled_image)
        minr, minc, maxr, maxc = region.bbox
        cropped_original = original[minr:maxr, minc:maxc]
        plot(cropped_original)
        try:
            item = int(input("what is this? \n1=bowl\n2=plate\n3=DROP\n4=ERROR\n"))
        except:
            item = 4
        # item = int(input("what is this? \n1=bowl\n2=plate\n3=small_plate\n4=bread_bowl\n5=cup\n6=DROP\n7=ERROR\n"))
        return item
    else:
        return -1  # default value that is obviously not a classification


'''wraps plt.imshow and plt.show for a one line work around to common python error in imshow()
@:param image is the image to show
'''
def plot(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


'''creating a panda frame representing the region (or an empty frame if False)
@:param original, original photo used to display to user in training mode
@:param region, the region to be captured. 
The captured data will be in the headers
["Area", "Orientation", "BBoxX", "BBoxY", 'Type_o_Object']
@:return the created data frame
'''
def create_pd_frame(original, region=False):
    if region is False:
        frame = pd.DataFrame(columns=["Area", "Orientation", "BBoxX", "BBoxY", 'Type_o_Object'])
    else:
        minr, minc, maxr, maxc = region.bbox
        frame = pd.DataFrame([[region.area, region.orientation, maxr - minr,
                               maxc - minc, label_obj(region, original, train=TRAIN)]],
                             columns=["Area", "Orientation", "BBoxX", "BBoxY", "Type_o_Object"])
        frame = frame[frame.Type_o_Object != 6]
    # print(frame)
    return frame


''' takes in a "regioned" image and for each region in the image
@:param image, the regioned image that contains the regions whose data will be captured
@:param original, the original non-regioned image used for training and displaying
@:param show a boolean whether or not show the image after processing
@:return the data of the regions in image joined
'''
def process_regions(image, original, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    data = create_pd_frame(original)
    # useful link == https://au.mathworks.com/help/images/ref/regionprops.html
    # print(measure.regionprops(image)[0])
    for region in measure.regionprops(image):
        # take regions with large enough areas
        if region.area >= MIN_AREA:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatch.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            region_data = create_pd_frame(original, region=region)
            data = data.append(region_data, ignore_index=True)
    # ax.set_axis_off()
    if show:
        plt.tight_layout()
        plt.show()
    return data

''' seperated an image into regions and removes the edge touching regions (strips it)
@:param image the image to be processed
@:return the image with regions
'''
def sep_and_strip_img(image):
    # turn the image into an image that has distinct regions
    image_labels, segments = measure.label(image, background=1, return_num=True)

    # remove artifacts connected to image border
    cleared_labeled = skimage.segmentation.clear_border(image_labels)
    return cleared_labeled


''' applys a binary filter based on saturation to an image
@:param image the image to be processed
@:return the image that is now a binary (on or off) mask
'''
def get_mask(image):
    image = rgb2gray(image)
    val = filters.threshold_otsu(image)
    mask = image < (val + MASK_SENSITIVITY)
    return mask


''' top level function to stitch together image manipulation 
to have the end level effect of converting image to data frame of the contained regions (objects of interest)
@:param img the image to be captures by data
@:return the data of the objects of interest
'''
def img_to_data(img):
    data = process_regions(sep_and_strip_img(get_mask(img)), img)
    return data


''' 
main function for isolating masks_obj features for debugging
'''
def masks_main():
    image = skio.imread("./sample_photos/20180923_095828.jpg")
    plot(image)
    plot(get_mask(image))
    plot(sep_and_strip_img(get_mask(image)))
    process_regions(sep_and_strip_img(get_mask(image)), image, show=True)
    print("data =\n" + str(img_to_data(image)))


# masks_main()



