"""
image classification using keras resnet50
copilot
"""

import numpy as np
from keras.applications import ResNet50
from matplotlib import pyplot as plt
from scipy.constants import lb
from sklearn.metrics import classification_report
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.layers.core import Flatten, Dense, Dropout

from src.helpers.clr_callback import CyclicLR

args = {}
data = []
labels = []

model = ResNet50(
    weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# construct our model using a	fully-connected	network
x = Flatten()(model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(len(lb.classes_), activation="softmax")(x)

# construct the model using our custom classifier
model = Model(inputs=model.input, outputs=x)

# train the model using SGD
print("[INFO] training model...")
opt = SGD(
    lr=1e-4, decay=1e-4 / args["epochs"], momentum=0.9,
    nesterov=True
)
model.compile(
    loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"]
)
H = model.fit(
    data, labels, validation_split=0.2, epochs=args["epochs"],
    callbacks=[CyclicLR(
        base_lr=1e-4, max_lr=1e-2,
        step_size=args["epochs"] // 2
    )]
)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(data, batch_size=32)
print(
    classification_report(
        labels, predictions.argmax(axis=1),
        target_names=lb.classes_
    )
)

# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# plot the learning rate history
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["lr"], label="learning_rate")
