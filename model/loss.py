from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
# Custom loss function that applies MSE specifically to RGB channels
def mse_rgb(y_true, y_pred):
    # y_true and y_pred are assumed to be in shape (batch_size, height, width, 3)
    return K.mean(K.square(y_true - y_pred), axis=-1)