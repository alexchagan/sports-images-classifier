import os
from dotenv import load_dotenv
import warnings
from tensorflow.python.client import device_lib

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
os.chmod(".kaggle/kaggle.json", 600)

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

print("Tensorflow is running on following devices : ")
print(device_lib.list_local_devices())