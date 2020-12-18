# udacity_dog_breed
Udacity capstone project to identify dog breeds

## Project Motivation
The goal of this project is to build models to identify if an image has a human or a dog and then predict which breed of dog was identified.  In the case of a human being identified, predict which breed of dog the human most closely resembles.  An algorithm that ties it all together could be used by an app to identify dog breeds based on images.  In all there will be 2 models, one for detecting humans in images and one for detecting dogs in images.  There will also be a convolutional neural network to predict the breed of dog found in the image.

Udacity provided a workspace in which to use the Jupyter notebook along with some helpful starter instructions and code snippets.  By using the Udacity workspace, no environment setup was required and a GPU was available to speed things up.

For more details check out the blog post: https://alexanli-wk.medium.com/cool-dog-i-wonder-what-kind-it-is-8833470767cd

### Udacity provided the files related to dog and human images.  
* Train, Validation, and Test dog images were provided in the form of paths to images
* Train, Validation, and Test dog target files were provided containing one-hot encoded classification tables
* Dog_names were provided as an array of strings to translate labels into something more easily understood
* There are 133 dog categories
* 8351 dog images were provided and split into train, validation, and test sets for model building
* 13233 human images were provided

### Human image classifier
* A pre-trained OpenCV Haar feature-based cascade classifier was used for the human face detection
* Documentation for OpenCV can be found here: https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
* This is the classifier used for the project: haarcascades/haarcascade_frontalface_alt.xml
* Images are converted to Grayscale before being sent to the classifier

### Dog image classifier
* A pre-trained ResNet-50 model was used to detect dogs in images
* This is an image of the ResNet-50 model: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
* The ResNet-50 model was trained on the ImageNet dataset - http://www.image-net.org/
* ImageNet contains over 10 million URLs linking to an object from 1000 categories
* Images are converted to a standard 224x224 size
* Images are then converted to a 3D tensor 224x224x3 to handle the color
* Images are then converted to a 4D tensor suitable for keras - n_sample x 224 x 224 x 3
* keras provides a preprocess_input to handle the re-ordering of the color channels and a normalization step
* the ResNet-50 model's predict method is used to get the prediction of whether the image contains a dog or not
* Argmax is used to get the integer corresponding to the model's predicted object class
* ImageNet provides this dictionary to identify the predicted object class - https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
  * Dog breeds occur between indexes 151-268, inclusive, so if the predicted object class falls in this range a dog is detected

### Convolutional Neural Network (CNN)
* Build a CNN to determine the breed of dog when a dog is detected in an image or the breed a human looks like when a human is detected
* Rescale images for the CNN by dividing by 255
* Pre-process the files for keras as described in the Dog image classifier
* The minimum acceptable accuracy is 1% within 5 epochs
* Build a CNN using a number of convolutional layers, pooling layers, dense layers
* Adjust filters, kernel_size, strides, padding, and activation to improve accuracy
* Add dropout layers to reduce over-fitting
* Model output should be 133 categories to correspond with the number of dog breeds provided in the dog_names list
* Compile the model using the rmsprop optimizer and loss set to categorical_crossentropy and metrics set to accuracy
* When fitting the model, use checkpointing to save the the model with the best validation loss
* Check model accuracy using test data.  Accuracy = number correct predictions / total number of predictions
* Adjust and tune to improve accuracy - look at data augmentation

### Transfer Learning to train CNN
* Build a CNN based on a pre-trained model using Transfer Learning
* The minimum acceptable accuracy for the model is 60%
* Try multiple pre-trained models - VGG16, ResNet-50, Inception           
* Load the bottleneck features of the pre-trained model
* Split the bottleneck features into Train, Validation, and Test sets
* Add additional layers as desired, but end with a Dense layer with 133 categories and softmax activation

### Create an algorithm to predict dog breed
* Using the CNN with the highest accuracy, predict dog breeds given images
* Algorithm should determine if the image contains a human or a dog or neither
* If a dog is detected, use the CNN to predict the breed
* If a human is detected, use the CNN to predict the breed the human is most like
* If neither a dog or human is detected, return an error
* Supply a set of images and test the algorithm


## Installations
Jupyter notebook and helper files build using Python v3.8.3

### Libraries Included:
* cv2
* extract_bottleneck_features
* glob - glob
* matplotlib.pyplot
* numpy
* pandas
* PIL - ImageFile
* random
* keras.applications.resnet50 - ResNet50, preprocess_input, decode_predictions
* keras.callbacks - ModelCheckpoint
* keras.preprocessing.image - ImageDataGenerator
* keras.layers - Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
* keras.models - Sequential
* keras.preprocessing - image
* keras.utils - np_utils
* sklearn.datasets - load_files
* tqdm - tqdm


## File Descriptions
* dog-project.ipynb - A Jupyter Python notebook to build and train the models

## How to use
* Download the Jupyter notebook from Github
* Install any packages necessary to import the libraries listed
* Open the Jupyter notebook and review the Project
* Upload the images you want to work with to a my_images folder or adjust the path to the files as needed
* The final cell has a predictor that will loop through any images found in a my_images folder


## Authors, Acknowledgements, Etc
* Author:  Lincoln Alexander
* Acknowledgements:  Udacity made me do it
