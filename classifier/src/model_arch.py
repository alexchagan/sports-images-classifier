import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications.efficientnet import EfficientNetB3
from keras import regularizers
from keras.optimizers import Adamax
from src.metrics import Metrics

class ModelArchitecture:

    def __init__(self, image_size, num_of_classes, model_name, learning_rate):

        self._image_size = image_size
        self._num_of_classes = num_of_classes
        self._model_name = model_name
        self._learning_rate = learning_rate
        self._model = None

    def make_model(self): 

        image_shape=(self._image_size[0], self._image_size[1], 3)

        if self._model_name == 'MobileNetV3Small':
            base_model=tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg= 'created MobileNetV3  small model'

        elif self._model_name == 'MobileNetV3Large':
            base_model=tf.keras.applications.MobileNetV3Large(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg='created MobileNetV3 large model'

        elif self._model_name == 'EfficientNetV2B0':        
            base_model=tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max')
            msg='Created EfficientNetV2 B0 model' 

        elif self._model_name == 'EfficientNetV2B1':
            base_model=tf.keras.applications.EfficientNetV2B1(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max') 
            msg='Created EfficientNetV2 B1 model'

        elif self._model_name == 'EfficientNetV2B2':        
            base_model=tf.keras.applications.EfficientNetV2B2(include_top=False, weights="imagenet",input_shape=image_shape, pooling='max') 
            msg='Created EfficientNetV2 B2 model' 
        
        else:
            base_model = EfficientNetB3(include_top=False, weights='imagenet',pooling='max', input_shape=(self._image_size, self._image_size, 3))
            msg='Created EfficientNetV2 B3 model'

        base_model.trainable=True
        x=base_model.output
        x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
        x = Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                        bias_regularizer=regularizers.l1(0.006) ,activation='relu')(x)
        x=Dropout(rate=.4, seed=123)(x)       
        output=Dense(self._num_of_classes, activation='softmax')(x)
        self._model=tf.keras.Model(inputs=base_model.input, outputs=output)
        self._model.compile(Adamax(learning_rate=self._learning_rate), loss='categorical_crossentropy', metrics=['accuracy', Metrics.F1_score]) 
        msg=msg + f' with initial learning rate set to {self._learning_rate}'
        print(msg)
        return self._model
    
   