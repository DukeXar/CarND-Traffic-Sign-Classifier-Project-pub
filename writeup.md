#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dukexar/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the plain Python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set consisting of randmoly choosen images.

![Training examples by class][writeup/exploratory.png]

I have also build a bar chart showing how the data is distributed among the classes.
It is clearly visible that dataset is not balanced good enough and many classes are underrepresented, and after training the network
can be not good enough in recongnizing those underrepresented classes.

![Training examples by class][writeup/training_examples_by_class.png]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

I have seen that many approaches convert to the grayscale as a first step in normalization. Theoretically, color is more important for
humans than for machines, as it allows to remember signs more easily, and by-design all traffic signs should have enough information
on them to be recognized in grayscale, but in severals works: [Pierre Sermanet and Yann LeCun paper](http://www.academia.edu/download/31151276/sermanet-ijcnn-11.pdf), and
[Multi-column Deep Neural Networks for Image Classification](https://arxiv.org/pdf/1202.2745.pdf), it appeared that grayscaling is not the most
important step to achieve a higher accuracy (compared to network architecture), and if CNN would think that color is not important, it should
filter it out. So I decided to go with color images.

I will normalize the image data by applying the adaptive historgram filter, and scaling inputs to be in the range from -1 to 1.
Histogram normalization is required because input data contains images with low and high contrast, low and high lighting, and variance
of the input data is quite high, and may prevent optimizer from doing its job well. Next step is to scale, so that the mean is closer to zero.

Here are the results of calculating variance and mean on the pixel values of random sample of input data after applying different preprocessing
techniques.

Input          var=5270.396220635308, max=255, min=11, mean=94.33151041666666
Grayscaling Preprocessed 1 var=0.06647280603647232, max=1.0, min=0.0, mean=0.4533267915248871
Adaptive histogram Preprocessed 2 var=0.05973304063081741, max=1.0, min=0.0, mean=0.43016353249549866
Adaptive histogram + scaling Preprocessed 3 var=0.23893216252326965, max=1.0, min=-1.0, mean=-0.1396729201078415

Here is an example of a traffic sign image before and after normalization (input images, grayscaling, histogram, histogram and scaling).

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The input data was provided already split into the training, cross validation and test datasets.

After the first training of the model on the normalized data set, it was confirmed that the model was overfitting the input data set and was not performing well
on the cross validation data set. Specifically it was not recognizing underrepresented images well enough. That confirmed the need to balance the training
data set.

The is no much benefit in balancing the cross validation and test datasets as they are not participating in actual training process.

Additional data was generated by applying various geometric transformations: rotation, scaling, translation, and projection transformations.

The X code cell of the IPython notebook contains the code for augmenting the data set.

Here are the examples of original (first column) and augmented images (other columns):

![alt text][image3]

The distribution of the images by class is the following:

![alt text][image3]

The normalized and extended data sets were stored for further reuse during model training.

After loading, the training dataset is initially shuffled.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **X** cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x128   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 16x16x128 				|
| Dropout               | keep probability 0.9 |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x128   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 8x8x128 				|
| Dropout               | keep probability 0.9 |
| Max pooling           | 2x2 stride, outputs 8x8x128, connected to output of first max pool layer |
| Fully connected		| 400 neurons        									|
| Dropout               | keep probability 0.5 |
| Fully connected		| 43 neurons        									|


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

The first architecture was the classic LeNet architecture as it was great in characters recognition, and seemed as a good start. After 10 epochs it
achieved training accuracy of 0.977 (from 0.878 in the first epoch), and validation set accuracy of 0.929 (0.897 on the first epoch). This was interpreted as
the network is not big enough for the complexity of the input data.

After reading various articles, was quite keen on trying the multiscale architecture, where the main idea is that fully connected layer receives inputs
not just from the last convolutional layer, but from all of them. As long as every next convolutional builds upon previous and recongnizes more complex
objects, some details in the complex objects can be important for the classifier. For example, shape of the sign can be recognizes by the second layer of the
network, while simple lines from the first layer can indicate that it is the "Priority Road" sign. Also it worked well **TODO article**.

The filter size was choosen as 5x5, which seems to be the generic one. The number of filters were pretty arbitrary - 32 and 128. The classifier was choosen to have 400
neurons in the hidden layer.

Initial training showed that the model was able to overfit the training dataset, achieving accuracy of more than 0.99 in less than 10 epochs. But it was not able to generalize
on the validation dataset, providing validation accuracy around 0.92. To fight with overfitting, a L2 regularization was added, raising the validation accuracy to 0.97. Then the dropout layers
were added after each layer with different probabilities for each layer. The generic approach was to have little or no dropout for the convolution layers and drop 50% of the data in the
classifier layer.

The following combinations of filters in convolution layers, and neurons in fully connected layer were tested:

| Model name  | Architecture | Training accuracy | Validation accuracy | Comments |
| ---         | ---          | ---               | ---                 | ---      |
| ms9         | 32-128-400   | 0.99999  | 0.92000 | No dropout and regularization |
| ms10        | 32-128-400   | 0.99999  | 0.97959 | L2 regularization and dropout, L2_lambda=0.001 |
| ms10_2      | 32-128-200   | 0.99999  | 0.97891 | same |
| ms10_3      | 32-64-400    | 0.99988  | 0.98231 | same |
| ms10_4      | 32-64-200    | 0.99998  | 0.97800 | same |
| ms10_1      | 128-128-400  | 0.99998  | 0.98912 | same, learning rate 0.0001 |
| ms10_1.1    | 128-128-400  | 0.99874  | 0.97982 | same, L2_lambda=0.01 |

With the regularization it is taking more epochs to achieve the high training accuracy, compared to the initial approach (ms9).

It is visible from the table, that most configurations were able to overfit the training
dataset even with regularization and dropouts, although the difference between validation and training accuracy is not as huge as without.

Another way to avoid overfitting was try to simplify the model by reducing number of filters and neurons in the fully connected layer, but such models performed not as good
on the validation dataset, compared to more complex ones.

The model that performed best on the validation dataset was chosen as the final one.


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The configuration of the training parameters is located in the ... cell of the notebook, there it is possible to choose the maximum number of epochs to train, the batch size,
learning rate, vector of keep probabilities for the dropout layers (last item is for the fully-connected layer, others are for the convolutional layers), L2 lambda
for L2 regularization, patience parameter in number of epochs for early termination.
It also provides a way to configure the function to create the model and where the model training state should be stored.

To train the model, the Adam optimizer was chosen as it is quite efficient and easy to use. The model graph is constructed in cell .... Here I define the evaluation function,
which is used to output the accuracy after each epoch. The evaluation is done with disabling dropout logic, as otherwise its results would be random.

The training code is located in the ... cell of the notebook. The training code is able to resume the training from the epoch where it stopped the last time, it keeps track
of the model state that performed best on the validation data set, implements early termination if model did not improve for a configured number of epochs, reports and displays
the progress of training and validation accuracies to allow visually recognize whether model is overfitting or underfitting.

The training itself is executed in the ... cell of the notebook.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
