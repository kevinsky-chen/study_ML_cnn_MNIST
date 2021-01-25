from CNN_class import CNN
from keras.models import load_model
from keras.datasets import mnist
import tensorflow as tf

# !!!若電腦的gpu大小不夠大(eg. 跑CNN可能需要6GB)，限制memory_limit才可以正常運作(經測試為2GB)
# 參考網址: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Loading dataset
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

# Preprocessing testing data
testing_input = CNN.preProcessor()
testing_input_image = testing_input.normalize(testing_input.flatten(test_feature))
testing_input_label = testing_input.one_hot_encoding(test_label)    # default labels in one-hot encoding

# Loading pre-trained model
cnn_reload = CNN.model()

print("載入模型 Mnist_CNN_model.h5")
model = cnn_reload.load_all('Mnist_CNN_model.h5')

"""
# OR Loading pre-trained weights
cnn_reload.layers()
cnn_reload.training(testing_input_image, testing_input_label)
cnn_reload.load_weights("Mnist_CNN_weights.weight")
"""

# 載入架構的training method, 並 predict
prediction = cnn_reload.testing(testing_input_image, testing_input_label)
CNN.preProcessor.show_images_labels_predictions(test_feature, test_label, prediction, 15)

