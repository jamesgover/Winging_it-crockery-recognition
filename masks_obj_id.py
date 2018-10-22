import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from skimage import io as skio
import skimage
import skimage.segmentation
import os
from skimage import filters, measure
from collections import namedtuple
import pandas as pd
from skimage.color import rgb2gray
# plt.show()
TRAIN = False
MASK_SENSITIVITY = 0.15
MIN_AREA = 1000
DATA_FILE = './data/data.csv'  # the file that stores data
PHOTO_FOLDER = './data/photos'  # directory where program will find photos to use as test set

    # min_column(minc)  -------> max_column (maxc)
# min_row (minr)    |
#                   |
# max_row (maxr)   ╲╱
#  |-------->
#  |
#  |
# ╲╱

Region = namedtuple('Region', ['data', 'location', "trailing_edge"])


def set_train(train):
    '''
    simple function to let functions toggle the train variable
    '''
    TRAIN = train


def vid_area():
    '''
    simple function to let adjust pixel quantity for video quality
    '''
    MIN_AREA = 1000


def get_item_string(item):
    if item is 1:
        item = 'cup'
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
            item = int(input("what is this? \n1=cup\n2=plate\n3=small plate\n4=DROP\n5=ERROR\n"))
        except:
            item = 5
        # item = int(input("what is this? \n1=bowl\n2=plate\n3=small_plate\n4=bread_bowl\n5=cup\n6=DROP\n7=ERROR\n"))
        return item
    else:
        return -1  # default value that is obviously not a classification


def plot(image):
    '''wraps plt.imshow and plt.show for a one line work around to common python error in imshow()
    @:param image is the image to show
    '''
    plt.imshow(image, interpolation='nearest')
    plt.show()


def create_pd_frame(original=None, region=False):
    '''creating a panda frame representing the region (or an empty frame if False)
    @:param original, original photo used to display to user in training mode
    @:param region, the region to be captured.
    The captured data will be in the headers
    ["Area", "Orientation", "BBoxX", "BBoxY", 'Type_o_Object']
    @:return the created data frame
    '''
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


def process_regions(image, original, show=False):
    ''' takes in a "regioned" image and for each region in the image
    @:param image, the regioned image that contains the regions whose data will be captured
    @:param original, the original non-regioned image used for training and displaying
    @:param show a boolean whether or not show the image after processing
    @:return the data of the regions in image joined
    '''
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


def sep_and_strip_img(image):
    ''' seperated an image into regions and removes the edge touching regions (strips it)
    @:param image the image to be processed
    @:return the image with regions
    '''
    # turn the image into an image that has distinct regions
    image_labels, segments = measure.label(image, background=1, return_num=True)

    # remove artifacts connected to image border
    cleared_labeled = skimage.segmentation.clear_border(image_labels)
    return cleared_labeled


def get_mask(image):
    ''' applys a binary filter based on saturation to an image
    @:param image the image to be processed
    @:return the image that is now a binary (on or off) mask
    '''
    image = rgb2gray(image)
    val = filters.threshold_otsu(image)
    # print(val)
    mask = image < (0.501953125 + MASK_SENSITIVITY)
    return mask


def img_to_data(img):
    ''' top level function to stitch together image manipulation
    to have the end level effect of converting image to data frame of the contained regions (objects of interest)
    @:param img the image to be captures by data
    @:return the data of the objects of interest
    '''
    data = process_regions(sep_and_strip_img(get_mask(img)), img)
    return data


def decode(image):
    """Decodes datamatrix barcodes in `image`.
        Args:
            image: `numpy.ndarray`, `PIL.Image` or tuple (pixels, width, height)
        Returns:
            :obj:`list` of :obj:`Decoded`: The values decoded from barcodes.
    """
    set_train(False)
    region_list = []
    for region in measure.regionprops(sep_and_strip_img(get_mask(image))):
        if region.area >= MIN_AREA:
            # print("k")
            minr, minc, maxr, maxc = region.bbox
            rect = mpatch.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            region_data = create_pd_frame(region=region)
            region_list.append(Region(data=region_data, location=(minr, minc, maxr, maxc), trailing_edge=minr))

    return region_list


def masks_main(data_file=DATA_FILE, photo_folder=PHOTO_FOLDER):
    '''
    main function for isolating masks_obj features for debugging
    '''
    img = skio.imread("./data/frame.jpg")
    print(str(decode(img)))
    #plot(get_mask(img))
    #plot(sep_and_strip_img(get_mask(img)))
    return
    for filename in os.listdir(photo_folder):
        if filename.endswith(".jpg"):
            set_train(False)
            image = skio.imread(PHOTO_FOLDER + "\\" + filename)
            plot(image)
            plot(get_mask(image))
            plot(sep_and_strip_img(get_mask(image)))
            continue
        else:
            continue

if __name__ == "__main__":
    masks_main()

