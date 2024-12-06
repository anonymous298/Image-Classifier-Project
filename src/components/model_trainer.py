import sys
import os

from src.exception import CustomException
from src.logger import get_logger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

logger = get_logger('model-trainer')

class ModelTrainer:
    def __init__(self):
        pass

    def initiate_pretrained_model(self):
        '''
            This functions loads and returns the pretrained model

            Returns:
                loads and returns the pretrained model
        '''

        try:
            logger.info('Loading ResNet-50 Pretrained Model.')
            resnet = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(256,256,3)
            )

            logger.info('Pretrained model loaded successfully')

            logger.info('freezing the conv layers of pretrained model')
            for layer in resnet.layers:
                layer.trainable = False

            logger.info('All Layers freezed')

            return resnet

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def initialize_model_transfer_learning(self):
        '''
        This functions creates Neural Network architecture with transfer learning

        Returns:
            returns ANN model including pretrained model inside it with transfer learning
        '''

        try:
            resnet = self.initiate_pretrained_model()

            logger.info('Creating our own ANN base')

            model = Sequential([
                resnet,  # Our Pretrained CNN layers
                Flatten(),   # Flatteing down the image
                Dense(256, activation='relu'),  # Adding 256 Neurons to learn complex features
                Dropout(0.5), # Adding dropout to randomly of and on some neurons to avoid overfitting
                Dense(128, activation='relu'), # Adding 128 Neurons to learn even more complex features
                Dropout(0.3), # Adding dropout to randomly of and on some neurons to avoid overfitting
                Dense(1, activation='sigmoid')
            ])

            logger.info('Model created Successfully')

            logger.info('Compiling Our Model')

            adam = Adam()
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

            logger.info('Model Compiled Successfully')

            return model

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def start_training(self, train_generator, test_generator):
        '''
        This function start training of the model

        Parameters:
            train_generator: training data generator object.
            test_generator: testing data generator object.

        Returns:
            returns: trained model object for evaluation
        '''

        try:
            logger.info('Initiating Model Training')

            model = self.initialize_model_transfer_learning()
            
            logger.info('initializing callbacks')
    
            es_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            os.makedirs('model_logs', exist_ok=True)
            tb_callback = TensorBoard(log_dir='model_logs/training1', histogram_freq=1)

            logger.info('Starting fitting our model')

            model.fit(
                train_generator,
                validation_data=(test_generator),
                epochs=10,
                callbacks=[es_callback, tb_callback]
            )

            logger.info('Model Trained Successfully')

            return model

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)