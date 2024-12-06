import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import get_image_data_path

logger = get_logger('image-data-generator-object')

class ImageGeneratorObject:
    def __init__(self):
        pass

    def get_image_generator_object(self, train_transformation_object, test_transformation_object):
        '''
        This function will take ImageTransformation object and returns the Generator object.

        Parameters:
            train_tranformation_object: Train ImageDataGenerator class object
            test_tranformation_object: Test ImageDataGenerator class object

        Returns:
            train and test generator object
        '''

        try:
            logger.info('Creating ImageDataGenerator object')
            
            train_data_path, test_data_path = get_image_data_path()

            logger.info('Creating Generator object for training data')

            train_generator = train_transformation_object.flow_from_directory(
                train_data_path,
                target_size=(256, 256),
                batch_size=32,
                class_mode='binary'
            )

            logger.info('Creating Generator object for testing data')

            test_generator = test_transformation_object.flow_from_directory(
                test_data_path,
                target_size=(256,256),
                batch_size=32,
                class_mode='binary'
            )

            logger.info('Training and Testing Generator object created')

            return (
                train_generator,
                test_generator
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)