import numpy as np
import SimpleITK as sitk
import six
from radiomics import featureextractor

def read_dcm_series(dcm_dir):
    """
    Args:
        dcm_dir: Str. Path to dicom series directory
    Returns:
        sitk_image: SimpleITK object of 3D CT volume.
    """
    reader = sitk.ImageSeriesReader()
    series_file_names = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(series_file_names)
    sitk_image = reader.Execute()

    return sitk_image


def extract_feature_unit(sub_img, p, q, padding=2):
    """
    Args:
        sub_img: Numpy array. The tumor area defined by mask
        p,q: Int. The index of central pixel
        padding: Int. Number of pixels padded on each side after extracting tumor
    Returns:
        features_temp: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
    """
    # p and q are used to index the central pixel
    mask = np.copy(sub_img)
    mask[:, :] = 0
    mask[p - padding:p + padding + 1, q - padding:q + padding + 1] = 1
    img_ex = sitk.GetImageFromArray([sub_img])
    mask_ex = sitk.GetImageFromArray([mask])
    extractor = featureextractor.RadiomicsFeatureExtractor()
    radio_result = extractor.execute(img_ex, mask_ex)

    features_temp = {}
    features_temp["first"] = []
    features_temp["shape"] = []
    features_temp["glcm"] = []
    features_temp["gldm"] = []
    features_temp["glrlm"] = []
    features_temp["glszm"] = []
    features_temp["ngtdm"] = []
    for key, val in six.iteritems(radio_result):
        if (key.startswith('original_firstorder')):
            features_temp["first"].append(val)
        elif (key.startswith('original_shape')):
            features_temp["shape"].append(val)
        elif (key.startswith('original_glcm')):
            features_temp["glcm"].append(val)
        elif (key.startswith('original_gldm')):
            features_temp["gldm"].append(val)
        elif (key.startswith('original_glrlm')):
            features_temp["glrlm"].append(val)
        elif (key.startswith('original_glszm')):
            features_temp["glszm"].append(val)
        elif (key.startswith('original_ngtdm')):
            features_temp["ngtdm"].append(val)
        else:
            pass
    return features_temp