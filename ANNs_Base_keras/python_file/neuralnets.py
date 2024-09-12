# -*- coding: utf-8 -*-
"""
# Artificial Neural Networks using Keras
"""

# import basic modules
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

"""
## The Perceptron
- Mimic the biological Neuron to do intelligent taks!"""

# import necessary modules, load iris data, define a perceptron, train it, and make predictions
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # Iris setosa

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)  # predicts True and False for these 2 flowers

# print the output
y_pred

"""**Note:** The `Perceptron` becomes a `SGDClassifier` with `loss="perceptron"`, no regularization, and a constant learning rate equal to 1."""

# Build and train a Perceptron

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss="perceptron", penalty=None, learning_rate="constant", eta0=1, random_state=42)
sgd_clf.fit(X, y)

"""**Note:** When the Perceptron finds a decision boundary that properly separates the classes, it stops learning, means that the decision boundary is often quite close to one class."""

# plots the decision boundary of a Perceptron on the iris dataset

from matplotlib.colors import ListedColormap

a = -per_clf.coef_[0, 0] / per_clf.coef_[0, 1]
b = -per_clf.intercept_ / per_clf.coef_[0, 1]
axes = [0, 5, 0, 2]
x0, x1 = np.meshgrid(
    np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
    np.linspace(axes[2], axes[3], 200).reshape(-1, 1),)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)
custom_cmap = ListedColormap(['#9898ff', '#fafab0'])

plt.figure(figsize=(7, 3))
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris setosa")
plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-",
         linewidth=3)
plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="lower right")
plt.axis(axes)
plt.title("Decision Boundary")
plt.show()

"""### Activation functions
- Non-linear functions responsible to convert non-linearly seperable data linearly seperable
"""

# visuals of some activation functions, and their derivatives

from scipy.special import expit as sigmoid

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

max_z = 4.5
z = np.linspace(-max_z, max_z, 200)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot([-max_z, 0], [0, 0], "r-", linewidth=2, label="Heaviside")
plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
plt.plot([0, 0], [0, 1], "r-", linewidth=0.5)
plt.plot([0, max_z], [1, 1], "r-", linewidth=2)
plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, np.tanh(z), "b-", linewidth=1, label="Tanh")
plt.grid(True)
plt.title("Activation functions")
plt.axis([-max_z, max_z, -1.65, 2.4])
plt.gca().set_yticks([-1, 0, 1, 2])
plt.legend(loc="lower right", fontsize=13)

plt.subplot(122)
plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Heaviside")
plt.plot(0, 0, "ro", markersize=5)
plt.plot(0, 0, "rx", markersize=10)
plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
plt.plot(z, derivative(np.tanh, z), "b-", linewidth=1, label="Tanh")
plt.plot([-max_z, 0], [0, 0], "m-.", linewidth=2)
plt.plot([0, max_z], [1, 1], "m-.", linewidth=2)
plt.plot([0, 0], [0, 1], "m-.", linewidth=1.2)
plt.plot(0, 1, "mo", markersize=5)
plt.plot(0, 1, "mx", markersize=10)
plt.grid(True)
plt.title("Derivatives")
plt.axis([-max_z, max_z, -0.2, 1.2])
plt.suptitle("Visuals of some activation functions")
plt.show()

"""
## Regression MLPs

### Overview
- MLPs (Multilayer Perceptrons) can be used for regression tasks.
- For single value prediction (e.g., house price), one output neuron is required.
- For multivariate regression (e.g., predicting multiple values like 2D coordinates), one output neuron per dimension is needed.
- Example: Predicting the center of an object in an image (2 neurons for 2D coordinates) and bounding box (2 neurons for width and height) results in 4 output neurons.

### Example: MLPRegressor in Scikit-Learn
- **MLPRegressor** class can be used to build an MLP with multiple hidden layers.
- Example configuration:
  - 3 hidden layers with 50 neurons each.
  - Trained on the **California housing dataset** using `fetch_california_housing()`.
  - Standardization of input features is critical for gradient descent to work effectively.
- The model uses:
  - **ReLU activation** in hidden layers.
  - **Adam optimizer** to minimize the mean squared error (MSE).
  - **ℓ2 regularization**, controlled by the `alpha` hyperparameter.

### Evaluation
- The model achieves a validation **RMSE** of about **0.505**, comparable to a random forest classifier.

### Output Layer Activation Functions
- No activation function is used in the output layer (allows any value output).
- For positive-only outputs, use:
  - **ReLU** or **softplus** (a smooth variant of ReLU).
- For bounded outputs, use:
  - **Sigmoid** (range 0 to 1) or **tanh** (range –1 to 1).
- Note: **MLPRegressor** does not support activation functions in the output layer.

### Limitations and Alternatives
- **MLPRegressor** is convenient but has limited features.
- For more complex needs, switching to **Keras** is recommended.
  
### Loss Functions
- **MSE** (Mean Squared Error) is typically used for regression.
- Alternatives:
  - **Mean Absolute Error** (for datasets with many outliers).
  - **Huber loss**: A mix of MSE and MAE, less sensitive to outliers but allows faster convergence.

### Typical MLP Regression Architecture

| Hyperparameter         | Typical Value                                              |
|------------------------|------------------------------------------------------------|
| # hidden layers         | 1 to 5                                                     |
| # neurons per layer     | 10 to 100                                                  |
| # output neurons        | 1 per prediction dimension                                 |
| Hidden activation       | ReLU                                                       |
| Output activation       | None (or ReLU/softplus for positive outputs, sigmoid/tanh for bounded outputs) |
| Loss function           | MSE (or Huber for outliers)                                |
"""

# generating a MLP regression model, import data, train the model, make predictions, calculate error

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)

print("The root mean squared error is:", rmse)

"""## Classification MLPs
- Responsible for classification tasks!
"""

# loading data and modules, define model, train the model, and calculate accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000,
                        random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print("The accuracy rate is:", accuracy)

"""# Implementing MLPs with Keras
- Keras is TensorFlow’s high-level deep learning API: it allows to build, train, evaluate, and execute all sorts of neural networks

## Building an Image Classifier Using the Sequential API
- Using Keras to load the dataset

**Tasks:** start by loading the fashion MNIST dataset. Keras has a number of functions to load popular datasets in `tf.keras.datasets`.
"""

# import module, making trin, test, and validation set

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

"""**Note:** Training set contains 60,000 grayscale images, each 28x28 pixels."""

# training data shape
X_train.shape

"""**Note:** Each pixel intensity is represented as a byte (0 to 255)."""

# data type
X_train.dtype

"""**Task:** scale the pixel intensities down to the 0-1 range and convert them to floats, by dividing with 255."""

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

"""**Task:** plot an image using Matplotlib's `imshow()` function, with a `'binary'`color map."""

# ploting an image

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

"""**Note:** The labels are the class IDs (represented as uint8), from 0 to 9."""

# labels
y_train

"""Here are the corresponding class names:"""

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

"""So the first image in the training set is an ankle boot:"""

class_names[y_train[0]]

"""**Tasks:** take a look at a sample of the images in the dataset."""

# Samples from Fashion MNIST

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.suptitle("Samples from Fashion MNIST dataset")
plt.show()

"""### Creating the model using the Sequential API
- Classification MLP with two hidden layers
"""

# define a dense model

tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# clear the session to reset the name counters
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# model details including layer, names
model.summary()

# display the model's architecture
#tf.keras.utils.plot_model(model, "model_structure.png", show_shapes=True)

model.layers

hidden1 = model.layers[1]
hidden1.name

model.get_layer('dense') is hidden1

weights, biases = hidden1.get_weights()
weights

weights.shape

biases

biases.shape

"""### Compiling the model"""

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

"""This is equivalent to:"""

# this cell is equivalent to the previous cell
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

# how to convert class ids to one-hot vectors
tf.keras.utils.to_categorical([0, 5, 1, 0], num_classes=10)

"""**Note:** it's important to set `num_classes` when the number of classes is greater than the maximum class id in the sample."""

# how to convert one-hot vectors to class ids
np.argmax(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    axis=1
)

"""### Training and evaluating the model"""

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# important model parameters
history.params

print(history.epoch)

"""Learning curves: the mean training loss
   and accuracy measured over each epoch,
   and the mean validation loss and accuracy
   measured at the end of each epoch.
"""

import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")
plt.title("Keras learning curves")
plt.show()

# shift the training curve by -1/2 epoch

plt.figure(figsize=(8, 5))
for key, style in zip(history.history, ["r--", "r--.", "b-", "b-*"]):
    epochs = np.array(history.epoch) + (0 if key.startswith("val_") else -0.5)
    plt.plot(epochs, history.history[key], style, label=key)
plt.xlabel("Epoch")
plt.axis([-0.5, 29, 0., 1])
plt.legend(loc="lower left")
plt.grid()
plt.title("Keras learning curves")
plt.show()

# evaluating the model
model.evaluate(X_test, y_test)

"""### Using the model to make predictions"""

# predictions

X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = y_proba.argmax(axis=-1)
y_pred

np.array(class_names)[y_pred]

y_new = y_test[:3]
y_new

# Correctly classified Fashion MNIST images

plt.figure(figsize=(7.2, 3))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.suptitle("Correctly classified Fashion MNIST images")
plt.show()

"""## Building a Regression MLP Using the Sequential API
- California housing problem and tackle it using the same MLP as earlier, with 3 hidden layers composed of 50 neurons each, with keras
"""

# load and split the California housing dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# define, train, and evaluate a Regression MLP model
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

print("The root mean squared error is:", rmse_test)

y_pred

"""## Building Complex Models Using the Functional API
Not all neural network models are simply sequential. Some may have complex topologies. Some may have multiple inputs and/or multiple outputs.
"""

# reset the name counters and make the code reproducible
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# define the model

normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])

model.summary()

# set a training method, train and evaluate the model, make predictions

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)

"""**Task:** Send different subsets of input features through the wide or deep paths! Sending 5 features (features 0 to 4), and 6 through the deep path (features 2 to 7). Note that 3 features will go through both (features 2, 3 and 4)."""

tf.random.set_seed(42)

# sending features to different paths

input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])

# set a training method, prepare data, train and evaluate the model, make predictions
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
                    validation_data=((X_valid_wide, X_valid_deep), y_valid))
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))

"""Adding an auxiliary output for regularization:"""

tf.keras.backend.clear_session()
tf.random.set_seed(42)

# regularizations

input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])

# setting optimizer, and error functions
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer, metrics=["RootMeanSquaredError"])

# setting feature to layers, train the model
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid))
)

# evaluate the model
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results

y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

y_pred_tuple = model.predict((X_new_wide, X_new_deep))
y_pred = dict(zip(model.output_names, y_pred_tuple))

"""
## Subclassing API for Dynamic Models
- The **Subclassing API** provides a more **imperative** programming style.
- Suitable for models that require dynamic behavior or more flexible code."""

# define model

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

tf.random.set_seed(42)  # extra code – just for reproducibility
model = WideAndDeepModel(30, activation="relu", name="my_cool_model")

# set optimizer, train and evaluate the model, make predictions
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=optimizer,
              metrics=["RootMeanSquaredError"])
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)))
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

"""## Saving and Restoring a Model"""

# delete the directory, in case it already exists

import shutil

shutil.rmtree("my_keras_model", ignore_errors=True)

model.save("my_keras_model", save_format="tf")

# contents of the my_keras_model/ directory
from pathlib import Path
for path in sorted(Path("my_keras_model").glob("**/*")):
    print(path)

# loading the saved model
model = tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

model.save_weights("my_weights")

model.load_weights("my_weights")

# list of my_weights.* files

for path in sorted(Path().glob("my_weights.*")):
    print(path)

"""## Using Callbacks
Specify a list of objects that Keras will call before and after training, before and after each epoch, and even before and after processing each batch.

- only save model when models performance on the validation set
"""

shutil.rmtree("my_checkpoints", ignore_errors=True)

# loading the already defined model, and trained it, and save best model on validation set
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints", save_weights_only=True)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb])

# another way to save best model on validation set
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=100,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[checkpoint_cb, early_stopping_cb])

# defining own custom callbacks for extra control
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

# using own custom callbacks
val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)),
    callbacks=[val_train_ratio_cb], verbose=0)

"""## Using TensorBoard for Visualization
- View the learning curves during training

Install `tensorboard-plugin-profile`:
"""

# installing
import sys
#if "google.colab" in sys.modules:
    #%pip install -q -U tensorboard-plugin-profile

shutil.rmtree("my_logs", ignore_errors=True)

"""definefunction that generates the path of the log subdirectory based on the current date and time,
   so that it’s different at every run.
"""

from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()

# builds the first regression model we used earlier
tf.keras.backend.clear_session()
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)

tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])

# details metadata, files
print("my_logs")
for path in sorted(Path("my_logs").glob("**/*")):
    print("  " * (len(path.parts) - 1) + path.parts[-1])

"""**Task:** `tensorboard` Jupyter extension and start the TensorBoard server"""

# Commented out IPython magic to ensure Python compatibility.
# Visualizing learning curves with TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir=./my_logs

"""**Note**: if you prefer to access TensorBoard in a separate tab, click the "localhost:6006" link below:"""

# display in seperate tab

if "google.colab" in sys.modules:
    from google.colab import output

    output.serve_kernel_port_as_window(6006)
else:
    from IPython.display import display, HTML

    display(HTML('<a href="http://localhost:6006/">http://localhost:6006/</a>'))

"""**Note:** visualize histograms, images, text, and even listen to audio using TensorBoard"""

test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)

        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)

        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)

        texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
        tf.summary.text("my_text", texts, step=step)

        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

"""Stopping this Jupyter kernel will automatically stop the TensorBoard server as well. Another way to stop the TensorBoard server is to kill it, if you are running on Linux or MacOSX. First, you need to find its process ID."""

# lists all running TensorBoard server instances

from tensorboard import notebook

notebook.list()

"""**Note:** Use the following command on Linux or MacOSX, replacing `<pid>` with the pid listed above:

    !kill <pid>

On Windows:

    !taskkill /F /PID <pid>

## Fine-Tuning Neural Network Hyperparameters
- Using the Fashion MNIST dataset to demonstate different effect of hyperparameters on the model
"""

# preparing the data

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# clear previous session and reproducibility
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# installing if needed
#if "google.colab" in sys.modules:
    #%pip install -q -U keras_tuner

# importing tuner and define model
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

# best tuner (hyperparameter)
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))

# top models
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
top3_params[0].values  # best hyperparameter values

# best trial
best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
best_trial.summary()

# best trial accuracy
best_trial.metrics.get_last_value("val_accuracy")

# training best model with full training dataset

best_model.fit(X_train_full, y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

"""builds the same model as before, with the same hyperparameters,
   but it also uses a Boolean normalize" hyperparameter to control
   whether or not to standardize the training data before fitting the model
"""

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)

# pass a tuner, instead of build_model function

hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_epochs=10, factor=3, hyperband_iterations=2,
    overwrite=True, directory="my_fashion_mnist", project_name="hyperband")

# take care of using a different subdirectory for each trial
root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10,
                       validation_data=(X_valid, y_valid),
                       callbacks=[early_stopping_cb, tensorboard_cb])

"""
This algorithm gradually learns which regions of the hyperparameter space are most promising by fitting
a probabilistic model called a Gaussian process.
"""
bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_trials=10, alpha=1e-4, beta=2.6,
    overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt")
bayesian_opt_tuner.search(X_train, y_train, epochs=4,
                          validation_data=(X_valid, y_valid),
                          callbacks=[early_stopping_cb])

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir {root_logdir}

"""**Note:** That was the full main concepts which can be used in many domains! Thank You!"""