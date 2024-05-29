import datetime
from src.utils import tr_plot, makefolder
import tensorflow as tf
from src.ask_callback import ASK
import mlflow
import mlflow.tensorflow
import numpy as np
import os
import tempfile


class Trainer:

    def __init__(self, epochs, train_gen, valid_gen, model, ask=False):
        self.epochs = epochs
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.model = model
        self.ask = ask

    def train(self):

        'Trains a CNN model based on the requested architecture'

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        log_dir = f'tb_callback_dir/{current_time}'
        makefolder('training_results')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='training_results/best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min',
        )
        rlronp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.4, patience=2, verbose=1, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0.0,
        )
        estop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=True,
        )
        callbacks = [rlronp, estop, tensorboard_callback, checkpoint_callback]

        if self.ask:
            ask_cb = ASK(model=self.model, epochs=self.epochs, ask_epoch=5)
            callbacks.append(ask_cb)

        with mlflow.start_run(run_name=f'image_classifier_{current_time}'):
            
            mlflow.set_tracking_uri('http://127.0.0.1:5000')

            mlflow.log_param('epochs', self.epochs)
            mlflow.log_param('model_name', self.model.name)
  
            with tempfile.TemporaryDirectory() as temp_dir:
               
                sample_data, sample_labels = next(self.train_gen)
                np.save(os.path.join(temp_dir, 'sample_train_data.npy'), sample_data[:100])
                np.save(os.path.join(temp_dir, 'sample_train_labels.npy'), sample_labels[:100])
                mlflow.log_artifact(os.path.join(temp_dir, 'sample_train_data.npy'), 'dataset')
                mlflow.log_artifact(os.path.join(temp_dir, 'sample_train_labels.npy'), 'dataset')
 
                sample_val_data, sample_val_labels = next(self.valid_gen)
                np.save(os.path.join(temp_dir, 'sample_val_data.npy'), sample_val_data[:100])
                np.save(os.path.join(temp_dir, 'sample_val_labels.npy'), sample_val_labels[:100])
                mlflow.log_artifact(os.path.join(temp_dir, 'sample_val_data.npy'), 'dataset')
                mlflow.log_artifact(os.path.join(temp_dir, 'sample_val_labels.npy'), 'dataset')
           
            history = self.model.fit(
                x=self.train_gen,
                validation_data=self.valid_gen,
                epochs=self.epochs,
                verbose=1,
                validation_steps=None,
                shuffle=True,
                callbacks=callbacks,
            )

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            print(acc, val_acc, loss, val_loss)
            tr_plot(history, 0)

            for epoch, (acc_val, val_acc_val, loss_val, val_loss_val) in enumerate(zip(acc, val_acc, loss, val_loss), 1):
                mlflow.log_metric('accuracy', acc_val, step=epoch)
                mlflow.log_metric('val_accuracy', val_acc_val, step=epoch)
                mlflow.log_metric('loss', loss_val, step=epoch)
                mlflow.log_metric('val_loss', val_loss_val, step=epoch)

            mlflow.tensorflow.log_model(self.model, f'image_classifier_model_{current_time}')

            self.model.save('training_results/final.h5')
            print('model was saved as: training_results/model.h5')
