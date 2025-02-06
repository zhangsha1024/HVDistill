from model.res16unet_hvdistill import *
from model.image_model_hvdistill import *
try:
    from model.res16unet import Res16UNet34C as MinkUNet
    from model.res16unet_hvdistill import Res16UNet34C as MinkUNet_Hvdistill
    from model.image_model_hvdistill import DilationFeatureExtractor_hvdistill as DilationFeatureExtractor_hvdistill

except ImportError:
    MinkUNet = None
try:
    from model.spconv_backbone import VoxelNet
except ImportError:
    VoxelNet = None
