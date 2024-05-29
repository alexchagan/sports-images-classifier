import global_vars as gv
import warnings
from tensorflow.python.client import device_lib
from data_handler import DataHandler
from trainer import Trainer
from model_arch import ModelArchitecture
from inference import Predictor
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
os.chmod(".kaggle/kaggle.json", 600)

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())


def data_preparation():
    """
    Prepare the datasets for training and testing.
    Returns:
        train_gen (DataGenerator): Training data generator.
        valid_gen (DataGenerator): Validation data generator.
        test_ds (tf.data.Dataset): Test dataset.
    """

    DataHandler.download_from_kaggle(
        os.getenv("DATASET_OWNER"), os.getenv("DATASET_NAME")
    )
    data_handler = DataHandler(batch_size=gv.BATCH_SIZE, image_size=gv.IMAGE_SIZE, balance_classes=True)
    data_handler.prepare_datasets()
    train_gen, valid_gen = data_handler.define_generators()
    test_ds = data_handler.get_test_dataset()
    return train_gen, valid_gen, test_ds


def model_preparation():
    """
    Prepare the model architecture.
    Returns:
        model (tf.keras.Model): Compiled model.
    """

    model_arch = ModelArchitecture(
        image_size=gv.IMAGE_SIZE,
        num_of_classes=gv.NUM_CLASSES,
        learning_rate=gv.LEARNING_RATE,
        model_name=gv.MODEL_NAME,
    )
    model = model_arch.make_model()
    return model


def train_model(train_gen, valid_gen, model):
    """
    Train the model and save it locally.
    Args:
        train_gen (DataGenerator): Training data generator.
        valid_gen (DataGenerator): Validation data generator.
        model (tf.keras.Model): Compiled model.
    """

    trainer = Trainer(train_gen=train_gen, valid_gen=valid_gen, model=model, epochs=gv.EPOCHS)
    trainer.train()


def inference_model(test_ds, model_path):
    """
    Perform inference on the test dataset using the trained model.
    Args:
        test_ds (tf.data.Dataset): The test dataset to perform inference on.
        model_path (str): The path to the trained model file.
    """

    predictor = Predictor(test_ds=test_ds, image_size=gv.IMAGE_SIZE, model_path=model_path)
    predictor.predict()


if __name__ == "__main__":

    train_gen, valid_gen, test_ds = data_preparation()
    model = model_preparation()
    train_model(train_gen, valid_gen, model)
    inference_model(test_ds, "training_results/best.h5")
