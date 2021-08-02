## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The objective of this project is to use deep convolutional neural networks to classif traffic signs. The model is trained using [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and tested on images of the German traffic signs from the web. 

Steps
---
The following steps are comsidered to accomplish the defined objective.

* Load, explore, transform, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

Implementation
---

1. **Loading, exploring, transforming, and summarizing data**: Basic yet relevant steps were implemented to explore the data for purposes of gaining insight regarding the nature of the data. Simple statistical summaries were performed whose outcome could possibly affect how we reason about model development including the application of advanced computational techniques to better expose the statisitical features of the images to the model while aiding the model to pick up right inductive biases. Class or category inbalance is among statistical properties of a dataset that affects model performance in classification tasks. The plot below shows distributions of traffic sign categories in the training data.

![img6](https://user-images.githubusercontent.com/76077647/127775679-90405c5b-b2d5-4106-b00a-e1a63b42c2b7.JPG)

The table shows a summary of the data:

description|measure
-----------|-----------
Train size|34,799
Test  size|12,630
Image shapes|(32, 32, 3)
Number of classes|43

A pipeline of image transformations were implemented. Using prior knowledge of image transformations that prove effective for classification purporses, we chose the following techniques:

* Scaling
* Warpiing
* Brightness
* Translation

These were implemented as function utilities that were applied in a pipeline to each individual image. The image  below shows a result after applying the translation utility.

![img7](https://user-images.githubusercontent.com/76077647/127776171-8cd0fb8e-431c-47dd-9dae-5e1fc4b32ef3.JPG)


```
# HELPER FUNCTIONS FOR DATA AUGMENTATION

import cv2

def _image_scaling(img):
    """Applies perspective transformation to a given image"""
    rows,cols,_ = img.shape
    # transform limits
    px = np.random.randint(-2,2)
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(rows,cols))
    dst = dst[:,:,np.newaxis]

    return dst

def _image_warp(img):
    """Applies warp affine transformation to a given image"""
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]

    return dst

def _image_brightness(img):
    """Manipulates brightness/contrast of an image"""
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

def _image_translate(img):
    rows,cols,_ = img.shape
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

def _image_visualize(img, transformation_function):
    test_dst = transformation_function(img)
    fig, axs = plt.subplots(1,2, figsize=(10, 3))

    axs[0].axis('off')
    axs[0].imshow(test_img.squeeze(), cmap='gray')
    axs[0].set_title('original')

    axs[1].axis('off')
    axs[1].imshow(test_dst.squeeze(), cmap='gray')
    axs[1].set_title('transformed')
```

3. **Designing, training, and testing the model**: Our model' architecture was inspired by **LeNet-5** due to Yann LeCun. The original architecture is shown below however, our implementation is a slight modification to fit our purpose.

![img8](https://user-images.githubusercontent.com/76077647/127776469-ba8fafe0-ccd3-4dcb-9bd2-c36ab37e3ded.JPG)

The code block implementing our variation of LeNet-5 looks as below:

```
def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x
    
    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)
    layer3 = x

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2flat = flatten(layer2)
    print("layer2flat shape:",layer2flat.get_shape())
    
    # Flatten x. Input = 1x1x400. Output = 400.
    xflat = flatten(x)
    print("xflat shape:",xflat.get_shape())
    
    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    x = tf.concat([layer2flat, xflat], 1)
    print("x shape:",x.get_shape())
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 800. Output = 43.
    W4 = tf.Variable(tf.truncated_normal(shape=(800, 43), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(43), name="b4")    
    logits = tf.add(tf.matmul(x, W4), b4)
    
    return logits
```
The table summarizes a few hyperparameters used to train our implementation.

Hyperparam|value
----------|---------
Learning rate|0.0009
Batch size|128
Epochs|50
Dropout|True*
Padding|VALID*

*The asterisk denotes where necessary*

5. **Making predicitons**: The model's performance proved superior. With 50 epochs, the model had attained an accuracy score of **99.2%** on the validation set. When tried on a set of German signs taken from the web, the model's performance was an impressive **100%** showing that model had picked up the right inductive biases to perform its inteded task. A point to note is that relevant callbacks to be applied to even further optimize the model - one such callback could be early stopping. Observe that the model had obtained higher accuracy score prior to the 50th epoch.

![img9](https://user-images.githubusercontent.com/76077647/127785105-09ac0d11-400a-4f5a-915e-cee646daaea2.JPG)

7. **Analysis of softmax probabilities of new images**: A detailed analysis of the model's softmax probability scores of what the test images (ones from the web) could have been, shows its exceptional confidence in its predictions of class/category membership of each individual image. The image below is taken from the output of the model's guesses showing the actual input image with related space of softmax scores from the model's predictions.

![img10](https://user-images.githubusercontent.com/76077647/127777353-ea1f3438-e0d7-4272-81c3-26fc38e8ccbe.JPG)


Conclusion
---

This goal of this project was to build a German traffic sign classifier using classic deep convolutional neural networks. Granted the success of LeNet-5, we used LeNet-5 as the foundational architecture to build our classifier model. The model demonstrated superior performance in classifying random German traffic signs extracted from the web. Future improvements could clonsider the use of techniques to regularize model's training. Earlier we noted that the model had scored an accuracy higher than the final reported, we could apply callbacks such as early stopping to aide in settling for a model at its optimum during training. Other techniques could involve transfer learning, whereby we could use a pretrained image model such as the VGG models and only fine tune output layers to fit to our application. There are a host of other techniques that we could consider but in the meantime, our current implemetation showcased impressive performance as is. 

References & Credits
---

* ![Udacity - Self-Driving Car NanoDegree](http://www.udacity.com/drive)
* https://github.com/dkarunakaran
