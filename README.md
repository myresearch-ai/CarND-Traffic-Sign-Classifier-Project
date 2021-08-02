## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The objective of this project is to use deep convolutional neural networks to classif traffic signs. The model is trained using [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and tested on random images of the German traffic signs from the web. 

Steps
---
The following steps are comsidered to accomplish the defined objective.

* Load, explore, transform, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

Implementation
---

1. **Loading, exploring, transforming, and summarizing data**: Basic yet relevant steps were implemented to explore the data for purposes of gaining insight regarding the nature of the data. Simple statistical summaries were performed whose outcome could possibly affect how we reason about model development including the application of advanced computational techniques to better expose the statisitical characteristics of the images to the model while allowing the model to pick up right inductive biases. Class or category inbalance is among statistical properties of a dataset that affects model performance in classification tasks. The plot below shows distributions of traffic sign categories in the training data.

![img6](https://user-images.githubusercontent.com/76077647/127775679-90405c5b-b2d5-4106-b00a-e1a63b42c2b7.JPG)

The table shows a summary of the data:

description|measure
-----------|-----------
Train size|34,799
Test  size|12,630
Image shapes|32x32x3
Number of classes|43

A pipeline of image transformations was implemented. Insipred by transformations that previously worked well on traffic sign classifications e.g LeCun, the following [OpenCV2](https://opencv.org/) wrapper functions were implemented following standard practice:

* Warping affine
* perspective Transform
* Brightness

These were implemented as function utilities that were applied in a pipeline to each individual image. The image  below shows a result after applying the translation utility.

![img7](https://user-images.githubusercontent.com/76077647/127776171-8cd0fb8e-431c-47dd-9dae-5e1fc4b32ef3.JPG)


```
# HELPER FUNCTIONS FOR IMAGE DATA TRANSFORMATIONS

def _image_warpaffine(img):
    """Applies warp affine transformation to a given image"""
    rows,cols = img.shape[0:2]

    # random scaling coefficients
    random_x = np.random.rand(3) - 0.5
    random_x *= cols * 0.04 
    random_y = np.random.rand(3) - 0.5
    random_y *= rows * 0.04

    # 3 starting points for transform
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    points_in = np.float32([[y1,x1],[y2,x1],[y1,x2]])
    points_out = np.float32([[y1+random_y[0],x1+random_x[0]],[y2+random_y[1],x1+random_x[1]],
                             [y1+random_y[2],x2+random_x[2]]])
    M = cv2.getAffineTransform(points_in,points_out)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]

    return dst

def _image_perspective(img):
    """Applies perspective tranformation to a given image"""
    # Specify desired outputs size
    width, height = img.shape[0:2]
    # Specify congugate x, y coordinates
    pixels = np.random.randint(-2,2)
    points_in = np.float32([[pixels,pixels],[width-pixels,pixels],[pixels,height-pixels],[width-pixels,height-pixels]])
    points_out= np.float32(([[0,0],[width,0],[0,height],[width,height]]))
    # Perform perspective transformation using cv2
    M = cv2.getPerspectiveTransform(points_in,points_out)
    dst = cv2.warpPerspective(img,M,(width,height))
    dst = dst[:,:,np.newaxis]
    
    return dst

def _image_brightness(img):
    """Manipulates brightness/contrast of an image"""
    shifted_img = img + 1.0   
    img_max_value = max(shifted_img.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted_img * coef - 1.0
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
Learning rate|0.0005
Batch size|256
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

This goal of this project was to build a German traffic sign classifier using classic deep convolutional neural networks. Granted the success of LeNet-5, we used LeNet-5 as the foundational architecture to build our classifier model. The model demonstrated impressive performance in classifying random German traffic signs extracted from the web. Future improvements could consider the use of callbacks to perform helpful actions such as early stopping to train more optimal models. Earlier we noted that the model had scored an accuracy higher than the final reported score, early stopping could have been useful. Other techniques could involve transfer learning, whereby we could use a scaled pre-trained image classifier such as the VGG models and only adapt the output layers to fit to our application. There are a host of other techniques that we could consider such as the use of genetic algorithms for model optimization but in the meantime, our current implemetation perfromed fairly well. 

References & Credits
---

* ![Udacity - Self-Driving Car NanoDegree](http://www.udacity.com/drive)
* https://github.com/dkarunakaran
