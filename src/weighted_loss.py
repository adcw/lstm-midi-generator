import tensorflow as tf
from keras.losses import Loss


class WeightedLoss(Loss):
    def __init__(self, regularization_factor=0.1, attribute_weights=None, name='weighted_loss'):
        super(WeightedLoss, self).__init__(name=name)
        self.regularization_factor = regularization_factor
        self.attribute_weights = attribute_weights if attribute_weights is not None else 1

    def call(self, y_true, y_pred):
        # Obliczanie straty podstawowej z uwzględnieniem wag atrybutów
        base_loss = tf.reduce_mean(tf.square(y_true - y_pred) * self.attribute_weights)

        # Obliczanie straty regularyzacji
        regularization_loss = self.regularization_factor * tf.reduce_sum(tf.abs(y_pred))

        # Obliczanie całkowitej straty
        total_loss = base_loss + regularization_loss

        return total_loss
