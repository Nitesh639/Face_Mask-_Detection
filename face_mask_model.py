# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os

# 1. now initialize the initial learning rate(INIT_LR)
# 2. number of epochs to train for(EPOCHS)
# 3. batch size(BS)

INIT_LR = 0.0001
EPOCHS = 10
BS = 32

DIRECTORY = r"dataset"                             # list of images in our dataset directory
CATEGORIES = ["with_mask", "without_mask"]        # in this directory have two folders with_mask(faces with mask), without_mask(faces without mask)


print("loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)                     # path of the images
    	image = load_img(img_path, target_size=(224, 224))     # now we resize the images in equal size of (224 , 224)
    	image = img_to_array(image)                            # now image converting into array
    	image = preprocess_input(image)                        # preprocess the image array
    	data.append(image)                                     # adding all array in list
    	labels.append(category)                                # and also category (in the string form)

# perform one-hot encoding on the labels

lb = LabelBinarizer()
labels = lb.fit_transform(labels)                              # transform into 0 & 1
labels = to_categorical(labels)                                # converting into binary values

data = np.array(data, dtype="float32")                          # ML model only deal with numpy array
labels = np.array(labels)                                       # so converting into numpy array

(trainX, testX, trainY, testY) = train_test_split(data, labels,      # splitting the data
	test_size=0.20, stratify=labels, random_state=50)                # 80% for training and 20% for testing

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,                                            # using this we have more data set
	width_shift_range=0.2,                                      # than our model give good accuracy
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(
						weights="imagenet",
						include_top=False,
						input_tensor=Input(shape=(224, 224, 3))
						)

# construct the head of the model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy",
			  optimizer=opt,
	          metrics=["accuracy"])


# train the head of the network
print("training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# serialize the model to disk
print("saving mask detector model...")
model.save("face_mask.model", save_format="h5")
print("Model Saved")

predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

