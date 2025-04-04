import tensorflow as tf
import tensorflowjs as tfj

model = tf.keras.models.load_model("D:/demo ai/emotion-backend/emotion_model.h5")
tfj.converters.save_keras_model(model, "D:/demo ai/demo ai mocker/public/emotion_model_tfjs")