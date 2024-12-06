import sys

from src.exception import CustomException
from src.logger import get_logger

from src.components.image_transformation_object import ImageTransformationObject
from src.components.image_data_generator import ImageGeneratorObject
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger('training-pipeline')

def main():
    logger.info('Training Pipe Started')

    image_transformation_object = ImageTransformationObject()
    train_transformation_object, test_transformation_object = image_transformation_object.get_image_data_object()

    image_gen_obj = ImageGeneratorObject()
    train_gen, test_gen = image_gen_obj.get_image_generator_object(train_transformation_object, test_transformation_object)

    model_trainer = ModelTrainer()
    model = model_trainer.start_training(train_gen, test_gen)

    model_evaluation = ModelEvaluation()
    model_evaluation.evaluate(train_gen, test_gen, model)

    logger.info('Training PipeLine Executed Successfully')