from tensorflow import keras
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet()

model = sm.Unet('resnet34', encoder_weights='imagenet')

model = sm.Unet('resnet34', classes=1, activation='sigmoid')


