import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models
from keras import datasets
import ssl


# Loading dataset
ssl._create_default_https_context = ssl._create_unverified_context
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalizing data into 0 to 255 (binary)
training_images, testing_images = training_images / 255, testing_images / 255

# Creating Classes
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Showing the first 16 training images 
for image in range(16):
    plt.subplot(4, 4, image + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[image], cmap= plt.cm.binary)
    plt.xlabel(class_names[training_labels[image][0]])

plt.show()

# Reducing amount of training/testing data going into our model (lowering CPU/GPU/Time in exchange of Accuracy)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Our CNN model being created, inputs of 32x32x3 (32 pixels x 32 pixels x 3 color channels (Red)(Green)(Blue))
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation= 'relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation= 'relu'))
model.add(layers.Dense(10, activation= 'softmax'))

model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])

# Training our CNN model and evaluating it on loss/accuracy
model.fit(training_images, training_labels, epochs= 10, validation_data= (testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Saving model so we don't have to re-train it later
model.save('CNN_Outdoors_image_classifier.model')

# Loading our trained CNN model
model = models.load_model('CNN_Outdoors_image_classifier.model')

# Importing a new image to be tested in our model
img = cv.imread('/Users/alexandergursky/Local_Repository/Datasets/Unstructured/CNN_Outdoors_image_classifier.model/frog.jpeg')

# Adjusting the color scheme from BGR to match the RGB color scheme our model uses
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap= plt.cm.binary)

# Predicting what the new images clasification will be
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")

