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

$$ {g}'(z) = 1 - {tanh(z)}^2 = 1 - {g(z)}^2 \tag{2} $$


* The **cost function** can be calculated from the following formula - 

$$ -\frac{1}{m} \sum\limits_{i = 1}^{m}(y^{(i)}\log\left(a^{(i)}\right) + (1-y^{(i)})\log\left(1- a^{(i)}\right)) \tag{3} $$

* So the formulas for **forward propagation** will looks like considering X as input and Y as output -  


    - Hidden Layer Input ->
$$  Z^{(1)} = W^{(1)}X + b^{(1)} \tag{4} $$

    - Hidden Layer OutPut through Activation(tanh) function ->
$$  A^{(1)} = g^{(1)}(Z^{(1)}) = tanh(Z^{(1)}) \tag{5} $$

    - Final Layer Input ->
$$  Z^{(2)} = W^{(2)}A^{(1)} + b^{(2)} \tag{6} $$

    - Final Layer Output through Sigmoid ->
$$  Y = A^{(2)} = g^{(2)}(Z^{(2)}) = sigmoid(Z^{(2)}) \tag{7} $$


* So the formulas for **back propagation** will looks like considering X as input and Y as output -


    - differential For Final Layer -> 
$$  dZ^{(2)} = A^{(2)} - Y \tag{8} $$

    - weight differentials for Final Layer -> 
$$  dW^{(2)} = dZ^{(2)} (A^{(1)})^{T} \tag{9} $$

    - bias differentials for final layer -> 
$$ db^{(2)} = dZ^{(2)} \tag{10} $$

    - differentials for hidden layer output -> 
$$ dZ^{(1)} = (W^{(2)})^{(T)}dZ^{(2)} * {g^{(1)}}'(Z^{(1)}) \tag{11}  $$ 

    - so from the formula (2) we can write 
$$ dZ^{(1)} = (W^{(2)})^{(T)}dZ^{(2)} * (1 - (g^{(1)}(Z))^{2}) = (W^{(2)})^{(T)}dZ^{(2)} * (1 - (A^{(1)})^{2}) \tag{12} $$

    - weight differentials for Hidden Layer ->
$$ dW^{(1)} = dZ^{(1)} (X^{(1)})^{T} \tag{13} $$

    - bias differentials for Hidden layer -> 
$$ db^{(1)} = dZ^{(1)} \tag{14} $$
