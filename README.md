# CNN-Outdoor-Objects
Outdoor Image Classification Using CNN Deep Learning

A Convolutional Neural Network is an image classification neural network which falls within a subset of machine learning called deep learning. Often deployed on unstructured data, in particular image-based data, Convolution Neural Networks operate by using multi-layered/dimensional inputs which are passed along to hidden layers (often called Neurons). These Neurons are interconnected with other Neurons which are either prior or post of itself. 
By using an activation function we can determine how “brightly” or “dimly” a passing object lights up said Neuron. This information is then passed along in sequence order to the adjoining Neuron's activation function, ultimately becoming classified on the highest accuracy in the group of output nodes.

The dataset that I was working with was the CIFAR-10 dataset which consist of 60,000 color images which are 32 by 32 pixels in size, with 6,000 images per class. The classes in the dataset are as follows: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is important to note that the classes are completely mutually exclusive, “trucks” in this dataset would be considered as heavy equipment vehicles. Pickup trucks are not represented in either the ‘truck’ nor ‘automobile’ class. Also, note that the dataset is clean in nature. The only cleaning that was done was for my testing images, where I had to use the photo editing software GIMP to scale down random google images into 32 by 32 pixels to apply to my trained model.

The premise of what I wanted to accomplish in this project was first and foremost to work with an algorithm that I have had little to no experience with, to test my abilities and expand into new machine learning concepts. Secondly, I wanted to create a neural network which could be integrated into a live camera system and used to interpret generic outdoor entities. Object detection is something very useful industry-wide and most people might not even have an idea that it is backed by a machine learning algorithm. Many industry applications for Convolutional Neural Network’s come to mind such as, pedestrian detection, workplace safety warnings, animal detection, quality control in manufacturing, threat detection, biometric authentication, and many more.

To create this Convolutional Neural Network, I used the ConvNet algorithm which is the standard CNN algorithm, meaning that it is the most generic. By ‘most generic’ I am conveying in terms of CNN’s that it is the simplest and first to come to mind. For example, one might hear the term “Regression Analysis” and instantly you might have Multiple Linear Regression come to mind. I believe that it is worth mentioning that there are many other algorithm’s that fall within CNN’s other than ConvNet such as ResNet, GoogleNet, VGGNet, ZFNet, LeNet, and AlexNet just to name a few of them.

The way that ConvNet CNN’s operate is by stacking the color channels red, green, and blue into a multi-dimensional object. With each layer being pre-processed by ‘pooling’ to reduce the scale of the object as it passes to an activation function inside the Neuron, with the activation function that I used being ReLU. Rectified Linear Unit Activation Function (ReLU) is the most common activation function because it is simple, does not saturate, and is not computationally expensive. After pooling and applying the activation function is accomplished, flattening takes place. Flattening is the term for stretching the multi-dimensional matrix object into a vector array for the next Neuron to pick up on. Towards the end of our CNN where our output nodes for classification are located, we will be applying the Softmax function which is logarithmic in nature. The reason for this, if you recall from earlier is that our activation function ReLU goes from 0 to infinity. We need to bin this linear array somehow into our predefined classes, Softmax is great for this.

To create and train this model I used Python version 3.9.4 along with the library packages: cv2 (Computer Vision 2, for working with images), NumPy (Numerical Python, for working with arrays), Matplotlib (Mathematical Plotting Library, to visualize our training images), Tensorflow (Pythons flagship library for working with machine learning algorithms), Keras (a deep learning library which works on top of Tensorflow), and ssl (Secure Sockets Layer, because I was having warnings trying to import the dataset).

I then imported the dataset CIFAR10 previously forementioned, from the Keras library. Where I simultaneously parsed the data into training and testing. Followed by normalizing the data into binary. Binary is a great normalization factor to use with image data, if we skip normalization the range of distribution of feature values would be different for each feature. Thus the learning rate would cause corrections in each dimension, differing from one another (proportionally speaking).

Next, we are going to do the fun part, creating and training our model! The first thing that I did was limit the number of images going into training and testing. From the 60,000 images that comes with the dataset natively, I reduced this to 20,000 for training and 4,000 for testing. The reason I did this was for the sake of computation time, however this lowers accuracy of the model. Secondly, we created the model as shown (Fig 6 below) and compiled the model using Adam which is a stochastic gradient descent optimizer. Sparse Cross Entropy as the loss functions optimizer, Sparse Cross Entropy is a loss function used in classification for if this object goes to this class or that class, probability wise. While fitting we set Epochs to 10 as we want the model to look at the images 10 times, Epochs is recursive, similar to the concept of K-Fold Cross Validation, but completely different. Epochs is iterative over the whole dataset where it sets how many times the model will review itself on the entire dataset. While K-Fold Cross Validation is parsing the data in sections and validating the batch on the fold that is left out.

Lastly, we saved our model so that we did not have to retrain it, followed by me loading the newly trained model back into the program. Now that we have a trained model, I decided to test the model on some photos that I found on google and preprocessed them to fit into our 32x32x3 dimension model. The accuracy I was working with going into this part of the program was around 73% but we also have to keep in mind that I reduced the training and validation data as well. The images I gathered are shown in Fig 7 and the remaining code is shown in Fig 8. The predictions for the 5 random photos I tested on the model were as follows: Frog = Frog, 
Car = Car, Horse = Horse, Deer = Deer, Plane = Ship. I found it very interesting that the model had no issues distinguishing between a Horse and a Deer but confused a Plane with a Ship. 
  
Appendix
  
Dataset  
https://www.cs.toronto.edu/~kriz/cifar.html
  
General CNN Information  
https://www.andreaperlato.com/aipost/cnn-and-softmax/
  
https://en.wikipedia.org/wiki/Convolutional_neural_network
  
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
  
https://www.ibm.com/cloud/learn/convolutional-neural-networks
  
https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529
  
Documentation  
https://www.tensorflow.org/api_docs
  
https://keras.io
  
https://pypi.org/project/opencv-python/
