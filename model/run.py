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

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
os.chmod(".kaggle/kaggle.json",600)

warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())
   
if __name__ == '__main__':

    DataHandler.download_from_kaggle(os.getenv('DATASET_OWNER'), os.getenv('DATASET_NAME'))

    data_handler = DataHandler(batch_size=gv.BATCH_SIZE, image_size=gv.IMAGE_SIZE, balance_classes=True)
    data_handler.prepare_datasets()
    train_gen, valid_gen = data_handler.define_generators()
    
    model_arch = ModelArchitecture(image_size=gv.IMAGE_SIZE, num_of_classes=gv.NUM_CLASSES, learning_rate=gv.LEARNING_RATE, model_name=gv.MODEL_NAME)
    model = model_arch.make_model()

    trainer = Trainer(train_gen=train_gen, valid_gen=valid_gen, model=model, epochs=gv.EPOCHS)
    
    trainer.train()

    # trained_model = keras.models.load_model('training_results/best.h5')

    # predictor = Predictor(test_ds=test_ds, image_size=gv.IMAGE_SIZE, model=trained_model)
    # predictor.predict()


