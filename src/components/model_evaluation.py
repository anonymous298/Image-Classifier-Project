import os 
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger('model-evaluation')

@dataclass 
class ModelSavingPath:
    model_path: str = os.path.join('model', 'model.h5')

class ModelEvaluation:
    def __init__(self):
        self.model_path = ModelSavingPath()

    def evaluate(self, train_generator, test_generator, model):
        '''
        This will evaluate our given model and saves our model

        Parameters:
            train_generator: training data generator object
            test_generator: testing data generator object
            model: our trained model object.

        Returns:
            returns: None

        Save:
            save our model to path
        '''

        try:
            logger.info('Started evaluating model')

            logger.info('Evaluating for train data')
            _, train_score = model.evaluate(train_generator)
            logger.info(f'Model Score On Training Data: {train_score}')
            print(f'Model Score On Training Data: {train_score}')

            logger.info('Evaluating for test data')
            _, test_score = model.evaluate(test_generator)
            logger.info(f'Model Score on Testing Data: {test_score}')
            print(f'Model Score on Testing Data: {test_score}')

            logger.info('Evaluation Completed')

            logger.info('Saving Our Model To Path')

            os.makedirs(os.path.dirname(self.model_path.model_path), exist_ok=True)

            model.save(self.model_path.model_path)
            logger.info('Model Saved')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)