import sys
import os

from src.exception import CustomException
from src.logger import get_logger

from keras_preprocessing.image import ImageDataGenerator


logger = get_logger('image_transformation_object')

class ImageTransformationObject:
    def __init__(self):
        pass

    def get_image_data_object(self):
        '''
        This functions return the ImageDataGenerator object from which we can create generator object

        Returns:
            train and test image data generator object.
        '''

        try:
            logger.info('Creating ImageDataGenerator transformation object')

            logger.info('Creating train_datagenerator object')
            train_datagen = ImageDataGenerator(
                fill_mode='nearest',
                horizontal_flip=True,
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2
            )

            logger.info('Creating test datagenerator object')
            test_datagen = ImageDataGenerator(
                rescale=1./255
            )

            logger.info('ImageDataGenerator Object created')

            return (
                train_datagen,
                test_datagen
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)