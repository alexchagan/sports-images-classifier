import datetime
from utils import tr_plot, makefolder
import tensorflow as tf
from ask_callback import ASK


class Trainer:

    def __init__(self, epochs, train_gen, valid_gen, model, ask=False):
        self.epochs = epochs
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.model = model
        self.ask = ask

    def train(self):
        """Trains a CNN model based on the requested architecture"""

        log_dir = "tb_callback_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        makefolder("training_results")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="training_results/best.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )
        rlronp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=2,
            verbose=1,
            mode="auto",
            min_delta=0.00001,
            cooldown=0,
            min_lr=0.0,
        )
        estop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=4,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        callbacks = [rlronp, estop, tensorboard_callback, checkpoint_callback]
        if self.ask:
            ask_cb = ASK(model=self.model, epochs=self.epochs, ask_epoch=5)
            callbacks.append(ask_cb)

        history = self.model.fit(
            x=self.train_gen,
            validation_data=self.valid_gen,
            epochs=self.epochs,
            verbose=1,
            validation_steps=None,
            shuffle=True,
            callbacks=callbacks,
        )

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        print(acc, val_acc, loss, val_loss)
        tr_plot(history, 0)

        self.model.save("training_results/final.h5")
        print("model was saved as: training_results/model.h5")
