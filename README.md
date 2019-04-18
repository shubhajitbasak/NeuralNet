# Neural Net
### A Deep Neural Network From Scratch Using Python

### Part I 
#### A Shallow Neural Network with One Hidden Layer

Our human brain consists of 100 billion neurons connected together, if these neurons got sufficient inputs are given to any individual neuron it will fire the next neuron. The same can be modeled and we can create a neural network where each node is similar to the neurons. We can assign weights and bias at each neurons and layers so that the signals can progress through the layers which is known as forward propagation. Then at the final layer we calculate the loss with respect to the expected label and pass the error backward to different nodes to penalise the weights and other parameters, this process is known as backward propagation. Then we will keep iterating this process to change the weights of different layers until convergence.

Following are the formulations and assumptions we made -

* Input X will be a matrix of size (No Of Attributes, No of examples)
* Output Y will be a matrix of size (1, No of Examples) - as we are only supporting one output node 
* We are constructing a single hidden layer shallow neural network  

* For the final layer we are using signoid function to get the softmax value  

* We are using **tanh** as the activation function for the hidden layer -

     ![](https://latex.codecogs.com/gif.latex?g%28z%29%20%3D%20tanh%28z%29%20%3D%20%5Cfrac%7Be%5Ez%20-%20e%5E%7B-z%7D%7D%7Be%5Ez%20+%20e%5E%7B-z%7D%7D)

  -  So through calculas we can show the derivative of tanh as -
  
  ![](https://latex.codecogs.com/png.latex?%7Bg%7D%27%28z%29%20%3D%201%20-%20%7Btanh%28z%29%7D%5E2%20%3D%201%20-%20%7Bg%28z%29%7D%5E2)


* The **cost function** can be calculated from the following formula - 

     ![](https://latex.codecogs.com/png.latex?-%5Cfrac%7B1%7D%7Bm%7D%20%5Csum%5Climits_%7Bi%20%3D%201%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%5Clog%5Cleft%28a%5E%7B%28i%29%7D%5Cright%29%20+%20%281-y%5E%7B%28i%29%7D%29%5Clog%5Cleft%281-%20a%5E%7B%28i%29%7D%5Cright%29%29)

* So the formulas for **forward propagation** will looks like considering X as input and Y as output -  


    - Hidden Layer Input ->
![](https://latex.codecogs.com/png.latex?Z%5E%7B%281%29%7D%20%3D%20W%5E%7B%281%29%7DX%20+%20b%5E%7B%281%29%7D)

    - Hidden Layer OutPut through Activation(tanh) function ->
![](https://latex.codecogs.com/png.latex?A%5E%7B%281%29%7D%20%3D%20g%5E%7B%281%29%7D%28Z%5E%7B%281%29%7D%29%20%3D%20tanh%28Z%5E%7B%281%29%7D%29)

    - Final Layer Input ->
![](https://latex.codecogs.com/png.latex?Z%5E%7B%282%29%7D%20%3D%20W%5E%7B%282%29%7DA%5E%7B%281%29%7D%20+%20b%5E%7B%282%29%7D)

    - Final Layer Output through Sigmoid ->
![](https://latex.codecogs.com/png.latex?Y%20%3D%20A%5E%7B%282%29%7D%20%3D%20g%5E%7B%282%29%7D%28Z%5E%7B%282%29%7D%29%20%3D%20sigmoid%28Z%5E%7B%282%29%7D%29)


* So the formulas for **back propagation** will looks like considering X as input and Y as output -


    - differential For Final Layer -> 
![](https://latex.codecogs.com/png.latex?dZ%5E%7B%282%29%7D%20%3D%20A%5E%7B%282%29%7D%20-%20Y)

    - weight differentials for Final Layer -> 
![](https://latex.codecogs.com/png.latex?dW%5E%7B%282%29%7D%20%3D%20dZ%5E%7B%282%29%7D%20%28A%5E%7B%281%29%7D%29%5E%7BT%7D)

    - bias differentials for final layer -> 
![](https://latex.codecogs.com/png.latex?db%5E%7B%282%29%7D%20%3D%20dZ%5E%7B%282%29%7D)

    - differentials for hidden layer output -> 
![](https://latex.codecogs.com/png.latex?dZ%5E%7B%281%29%7D%20%3D%20%28W%5E%7B%282%29%7D%29%5E%7B%28T%29%7DdZ%5E%7B%282%29%7D%20*%20%7Bg%5E%7B%281%29%7D%7D%27%28Z%5E%7B%281%29%7D%29)
    - so from the formula (2) we can write 
![](https://latex.codecogs.com/png.latex?dZ%5E%7B%281%29%7D%20%3D%20%28W%5E%7B%282%29%7D%29%5E%7B%28T%29%7DdZ%5E%7B%282%29%7D%20*%20%281%20-%20%28g%5E%7B%281%29%7D%28Z%29%29%5E%7B2%7D%29%20%3D%20%28W%5E%7B%282%29%7D%29%5E%7B%28T%29%7DdZ%5E%7B%282%29%7D%20*%20%281%20-%20%28A%5E%7B%281%29%7D%29%5E%7B2%7D%29)

    - weight differentials for Hidden Layer ->
![](https://latex.codecogs.com/png.latex?dW%5E%7B%281%29%7D%20%3D%20dZ%5E%7B%281%29%7D%20%28X%5E%7B%281%29%7D%29%5E%7BT%7D)

    - bias differentials for Hidden layer ->
   ![](https://latex.codecogs.com/png.latex?db%5E%7B%281%29%7D%20%3D%20dZ%5E%7B%281%29%7D)


### Part II 
#### Test with a simple dataset which is separable through a non-linear boundary

![](https://github.com/shubhajitbasak/NeuralNet/blob/master/Images/Circle.PNG)

### Part III
#### Test with More Complex Dataset 

* I will now test it with a more complex dataset 
* CIFAR-10 dataset of 10 images: https://www.cs.toronto.edu/~kriz/cifar.html
* Each image is of size 32x32, with 3 colour channels (Red, Green, Blue): 
To make it suitable for your shallow neural net, use just 1 of the colour channels or average all three to convert the images to greyscale; then convert it to vector of 1024 floats

### Part IV
#### Add Deep Neural Feature

#### Enhancement 

* I have changed the activation function from tanh to ReLU as ReLU is computationally less expensive and works much faster than tanh
* I have also added an additional hidden layer to the network to make the model as deep neural network
* While initializing the weight I have normalised them with respect to the previous layer weight
* Using ReLU and the normalization during weight initialization will reduce the chance of getting into the vanishing or exploding gradient issue
* Implementation of forward and back propagation will remain same from part I, only the tanh activation function is replaced by ReLU 
* As I have used RELU following is the implementation of RELU
    ![](https://latex.codecogs.com/png.latex?ReLU%28x%29%3Dmax%280%2Cx%29)
* Implementation of derivative of ReLU Function 
![](https://latex.codecogs.com/png.latex?%7Bf%7D%27%28ReLU%28x%29%29%20%3D%201%5C%3B%20if%5C%3B%20x%3E0%5C%3B%20else%5C%3B%200)


* So during back propagation I have passed the activation output when x is greater than zero
