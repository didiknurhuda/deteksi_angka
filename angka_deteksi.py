import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Muat dataset MNIST bawaan Keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalisasi (0-1) dan tambahkan dimensi channel (28, 28, 1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# 3. Bangun arsitektur CNN dengan Functional API
inputs = layers.Input(shape=(28, 28, 1), name="input_image")
x = layers.Conv2D(8, (3, 3), activation="relu", name="conv1")(inputs)
x = layers.MaxPooling2D((2, 2), name="pool1")(x)
x = layers.Conv2D(16, (3, 3), activation="relu", name="conv2")(x)
x = layers.MaxPooling2D((2, 2), name="pool2")(x)
x = layers.Flatten(name="flatten")(x)
x = layers.Dense(64, activation="relu", name="dense1")(x)
outputs = layers.Dense(10, activation="softmax", name="output")(x)

model = models.Model(inputs=inputs, outputs=outputs, name="MNIST_CNN")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 4. Latih model
print("\nMelatih model CNN...")
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
)

# 5. Evaluasi dan simpan model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi pada data uji: {test_acc * 100:.2f}%")

model.save("mnist_cnn.keras")
print("Model berhasil disimpan sebagai 'mnist_cnn.keras'")