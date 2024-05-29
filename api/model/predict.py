from pathlib import Path
import tensorflow as tf
from utils import F1_score, get_classes
import yaml
from PIL import Image
import io
import numpy as np

__version__ = '0-1-0'

BASE_DIR = Path(__file__).resolve(strict=True).parent

model = tf.keras.models.load_model(f'{BASE_DIR}/model-{__version__}.h5', custom_objects={"F1_score": F1_score})

classes = get_classes()

with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
target_size = tuple(config['image_size'])

async def predict_pipeline(image_data):
    ''' Preprocesses the image and predicts the class '''    
    # Preprocess
    image_bytes = await image_data.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    pred = model.predict(image_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)
    classes = get_classes()

    return classes[pred_class], confidence
