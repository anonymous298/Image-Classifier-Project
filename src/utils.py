import os
import sys

from dataclasses import dataclass

from src.logger import get_logger
from src.exception import CustomException

logger = get_logger('utils')

def get_image_data_path():
    '''
    This functions returns the data path of training and testing data

    Returns:
        train and test data path
    '''
    logger.info('Initiazling Data Paths')

    train_data_path: str = os.path.join('dataset', 'training_set')
    test_data_path: str = os.path.join('dataset', 'test_set')

    return (
        train_data_path,
        test_data_path
    )

