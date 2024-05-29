import tensorflow as tf

class Metrics:

    @staticmethod
    def F1_score(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            return f1_val