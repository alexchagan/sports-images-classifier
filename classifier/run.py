import src as src
from src.data_handler import DataHandler
from src.trainer import Trainer
from src.model_arch import ModelArchitecture
from src.inference import Predictor
import os
import yaml

def data_preparation(data_handler : DataHandler):
    """
    Prepare the datasets for training and testing.
    Returns:
        train_gen (DataGenerator): Training data generator.
        valid_gen (DataGenerator): Validation data generator.
        test_ds (tf.data.Dataset): Test dataset.
    """

    DataHandler.download_from_kaggle(os.getenv("DATASET_OWNER"), os.getenv("DATASET_NAME"))
    data_handler.prepare_datasets()
    train_gen, valid_gen = data_handler.define_generators()
    test_ds = data_handler.get_test_dataset()
    return train_gen, valid_gen, test_ds


def model_preparation(model_arch : ModelArchitecture):
    """
    Prepare the model architecture.
    Returns:
        model (tf.keras.Model): Compiled model.
    """

    return model_arch.make_model()
    

def train_model(trainer : Trainer):
    """
    Train the model and save it locally.
    """

    trainer.train()


def inference_model(predictor : Predictor):
    """
    Perform inference on the test dataset using the trained model.
    """

    predictor.predict()


if __name__ == "__main__":

    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    epochs = config['epochs']
    batch_size = config['batch_size']
    image_size = tuple(config['image_size'])
    num_of_classes = config['num_of_classes']
    model_name = config['model_name']
    learning_rate = config['learning_rate']

    data_handler = DataHandler(
        batch_size=batch_size, 
        image_size=image_size, 
        balance_classes=True
    )
    train_gen, valid_gen, test_ds = data_preparation(data_handler)

    model_arch = ModelArchitecture(
        image_size=image_size,
        num_of_classes=num_of_classes,
        learning_rate=learning_rate,
        model_name=model_name
    )
    model = model_preparation(model_arch)

    trainer = Trainer(
        train_gen=train_gen,
        valid_gen=valid_gen,
        model=model, 
        epochs=epochs
    )
    train_model(trainer)

    predictor = Predictor(
        test_ds=test_ds, 
        image_size=image_size, 
        model_path='training_results/best.h5'
    )
    inference_model(predictor)



   