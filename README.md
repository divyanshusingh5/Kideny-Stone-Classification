# Kideny-Stone-Classification
The goal of this project is to develop a deep learning model to classify kidney stones using the EfficientNet architecture.The final output of the project will be a trained deep learning model that can accurately classify kidney stones from images.

Data Used

![image](https://user-images.githubusercontent.com/96836586/224798052-2a249014-ff6c-4423-affa-712a26f3c1d1.png)

The resulting dataset contains 12,446 unique data points, with cysts accounting
for 3,709, normal findings for 5,077, stones for 1,377, and tumors for 2,283. To
ensure ease of use, the data was split into train (80%), test (10%), and validation
(10%) datasets. This dataset provides a comprehensive overview of
kidney-related diagnoses, and can serve as an important resource for researchers
and healthcare professionals alike.


2.2 Data Pre-processing and Data Augmentation
1. Image Resizing: In order to ensure that our final model, Efficient-net 3,
can accurately analyze the images we feed it, we need to resize all images
to a common size of 224x250 pixels. This is necessary because the model is
designed to work with images of a specific size and format. By
standardizing all images to the same dimensions, we can ensure that the
model can accurately compare and analyze them. This step is crucial for
achieving accurate and reliable results from our image analysis. Therefore,
we need to pay close attention to this step to ensure the quality of our final
output.
2. Image Normalization: Batch Normalizing was done to reduce the internal
covariance shift, which is the phenomenon of the distribution of each
layer's inputs changing during training. This is a problem because a neural
network must learn the correct weights to produce an accurate output, and
if the distribution of inputs is continually changing, the network won't be
able to learn the correct weights. 
3. Image Augmentation: We take in a data frame, maximum samples,
minimum samples, and a column as parameters. Then, it groups the data
frame by the column specified and checks if the number of samples in any
group is greater than the maximum number of samples specified. If so, it
randomly samples the group to the maximum samples specified. It also
checks if the number of samples in any group is less than the minimum
number of samples specified. 
4. Image Filtering: Removing any unwanted noise from the images by
applying filters such as median filter, Gaussian filter, sobel filter, etc., is an
important step in ensuring that the data is accurate and reliable for further
analysis. It also helps to ensure that all of the data is of uniform quality,
that all of the samples are of the same size and shape, and that all the
classes specified are present in the trimmed data frame. 
5. Feature Extraction: In order to extract features from the images, we need
to first preprocess them. This involves resizing the images, normalizing the
colors, and converting them to a suitable format for use in the model. Once
the images have been preprocessed, we can extract important features
such as shape, color, texture, etc.By leveraging these more complex features, we can further enhance
the accuracy of the model.
6. Data Preparation:
This process of preparing the data set for training the model often requires
a range of operations, including but not limited to, label encoding, one-hot
encoding, feature scaling, normalization, and feature selection. Label
encoding is the process of mapping a given set of categorical data labels to
numerical values, and one-hot encoding is the process of creating a new
feature for each unique value in a given categorical feature. Feature scaling,
meanwhile, is the process of normalizing the range of values in a given
feature, while normalization is the process of scaling the data to a specific
range. Finally, feature selection is the process of selecting the most
important features when training a model, which are often determined by
the type of problem being solved.


Why Efficient-Net 3?



An empirical study has revealed that a balanced network width/depth/resolution
can be achieved by scaling each of them with a constant ratio. In light of this
observation, Efficient-Net B0 was created to be a highly beneficial yet
straightforward compound scaling method for medical image datasets. This is an
improvement over existing convolutional neural networks (CNNs) such as ResNet
and Inception. The main concept of Efficient-Net is to leverage a combination of
depthwise separable convolution, Squeeze-and-Excitation (SE) blocks, and a
novel scaling algorithm to construct an efficient neural network architecture. The
depthwise separable convolution is a form of convolution that divides a regular
convolution into two distinct operations: a depthwise convolution and a
pointwise convolution. The depthwise convolution is used to learn spatial
features while the pointwise convolution is used to analyze channel-wise
features. The Squeeze-and-Excitation block is an attention mechanism which
amplifies important features and suppresses unimportant features to increase
the network parameter efficiency. The novel scaling algorithm of Efficient-Net is
based on the notion that the network should use fewer parameters as the input
size increases. This scaling algorithm adjusts the network up or down depending
on the input size, thus allowing for more efficient networks without sacrificing
accuracy. All in all, Efficient-Net is an incredibly efficient neural network.

Steps To Inhance Image 


Resolution Scaling


![image](https://user-images.githubusercontent.com/96836586/224801251-5c460e52-ac57-4cfd-ab29-5c1bcc8157bd.png)



Width Scaling


Width scaling is an important component of EfficientNet, a family of
state-of-the-art convolutional neural network architectures. It is a technique that
automatically adjusts the width and depth of a model to maximize its accuracy.
Width scaling works by increasing the number of channels for each convolutional
layer in the network. This allows the model to learn more complex patterns,
leading to increased accuracy. The idea behind width scaling is to adjust the
model’s parameters in order to find the optimal combination of width and depth
that maximizes its accuracy.


![image](https://user-images.githubusercontent.com/96836586/224801562-4034bb49-1510-479d-98fc-8be3e0a37013.png)




Depth Scaling


Depth Scaling of EfficientNet is a method used to improve the accuracy of a deep
neural network. This method is based on the observation that deeper networks
are more accurate than shallower ones. The idea behind this approach is to use a
pre-trained model and then add additional layers to it. The number of layers
added can be determined based on the desired accuracy. This approach allows for
the network to become more accurate without having to increase the number of
parameters.


![image](https://user-images.githubusercontent.com/96836586/224801805-e9169a4f-0c59-4cd3-8cae-6ba314cde888.png)


