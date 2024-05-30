import numpy as np
from skimage.morphology import remove_small_objects


# def min_max_normalize(volume):
#     volume = (volume - np.min(volume)) / (np.max(volume)-np.min(volume)) * 255.0
#     return volume.astype(np.uint8)
def min_max_normalize(image, low_perc=0.05, high_perc=99.95):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32)


def z_score_normalize(volume, mean=14149.058367563766, std=14855.695311585796):
    volume = (volume - mean) / std
    return volume.astype(np.float32)


def pad_background(image, dim2pad=(224, 224, 224)):
    """
    to invert the operation, use :
    inverted_image = np.zeros_like(image)
    inverted_image[crop_index] = padded_image[padded_index]
    """

    # use np.nonzero to find the indices of all non-zero elements in the image
    nz = np.nonzero(image)

    # get the minimum and maximum indices along each axis
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)

    # crop the image to only include non-zero values
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    cropped_img = image[crop_index]
    padded_image = np.zeros(dim2pad)

    # crop further if any axis is larger than dim2pad
    crop_index_new = crop_index
    if cropped_img.shape[0] > dim2pad[0]:
        cx, cx_pad = cropped_img.shape[0] // 2, dim2pad[0] // 2
        cropped_img = cropped_img[cx - cx_pad : cx + cx_pad, :, :]
        crop_index_new = (
            slice(crop_index[0].start + cx - cx_pad, crop_index[0].start + cx + cx_pad),
            crop_index[1],
            crop_index[2],
        )
    if cropped_img.shape[1] > dim2pad[1]:
        cy, cy_pad = cropped_img.shape[1] // 2, dim2pad[1] // 2
        cropped_img = cropped_img[:, cy - cy_pad : cy + cy_pad, :]
        crop_index_new = (
            crop_index_new[0],
            slice(crop_index[1].start + cy - cy_pad, crop_index[1].start + cy + cy_pad),
            crop_index_new[2],
        )
    if cropped_img.shape[2] > dim2pad[2]:
        cz, cz_pad = cropped_img.shape[2] // 2, dim2pad[2] // 2
        cropped_img = cropped_img[:, :, cz - cz_pad : cz + cz_pad]
        crop_index_new = (
            crop_index_new[0],
            crop_index_new[1],
            slice(crop_index[2].start + cz - cz_pad, crop_index[2].start + cz + cz_pad),
        )

    # calculate the amount of padding needed along each axis
    pad_widths = [(padded_image.shape[i] - cropped_img.shape[i]) // 2 for i in range(3)]

    # pad the image with zeros
    padded_index = tuple(slice(pad_widths[i], pad_widths[i] + cropped_img.shape[i]) for i in range(3))
    padded_image[padded_index] = cropped_img

    return padded_image, crop_index_new, padded_index


def invert_padding(original_image, padded_image, crop_index, padded_index):
    # crop the padded image to the size of the original image
    cropped_img = padded_image[padded_index]

    # create an array of zeros with the same shape as the original image
    inverted_image = np.zeros_like(original_image)

    # insert the cropped padded image into the center of the array of zeros
    inverted_image[crop_index] = cropped_img

    return inverted_image


def pad_background_with_index(image, crop_index_new, padded_index, dim2pad=(256, 256, 256)):
    padded_image = np.zeros(dim2pad)
    crop_image = image[crop_index_new]
    padded_image[padded_index] = crop_image
    return padded_image


def remove_small_elements(segmentation_mask, min_size_remove=3):
    # Convert segmentation mask values greater than 0 to 1
    pred_mask = segmentation_mask > 0
    # Remove small objects (connected components) from the binary image
    mask = remove_small_objects(pred_mask, min_size=min_size_remove)
    # Multiply original segmentation mask with the mask to remove small objects
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation
