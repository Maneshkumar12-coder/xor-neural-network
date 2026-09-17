Neural Network from Scratch: Solving XOR

A simple neural network built from scratch using Python and NumPy to learn the XOR problem.

This project was made to understand how a basic neural network actually works instead of using ready-made machine learning libraries.

About the Project

The XOR problem is a simple classification problem, but it cannot be solved by a single linear layer. Because of this, it is a good example for learning how hidden layers and nonlinear activation functions help a neural network learn more complex patterns.

In this project, I implemented a small neural network myself using only NumPy.

The network uses:

- Forward propagation
- Backpropagation
- Gradient-based weight updates
- Sigmoid activation
- Bias terms
- Mean Squared Error (MSE)
- Gradient descent

No TensorFlow, PyTorch, or scikit-learn was used.

Network Structure

The network has three layers:

Input Layer       Hidden Layer       Output Layer

   2 neurons  →      4 neurons   →     1 neuron

The two input values represent the XOR input, and the output represents the predicted result.

XOR Dataset

Input| Expected Output
[0, 0]| 0
[0, 1]| 1
[1, 0]| 1
[1, 1]| 0

How It Works

1. Forward Propagation

The input is first multiplied by the weights and added to the bias.

For the hidden layer:

h = sigmoid(XW₁ + b₁)

Then the hidden layer produces the final output:

output = sigmoid(hW₂ + b₂)

2. Sigmoid Function

The network uses the sigmoid activation function:

sigmoid(x) = 1 / (1 + e⁻ˣ)

It converts the values into a range between 0 and 1, which works well for this binary XOR example.

Its derivative is:

sigmoid'(x) = x(1 - x)

3. Calculate the Error

The difference between the expected output and the prediction is calculated:

error = y - output

The project also calculates the Mean Squared Error during training:

MSE = mean((y - output)²)

4. Backpropagation

Backpropagation is used to calculate how much each weight and bias contributed to the error.

The calculated values are then used to update the network parameters.

5. Training

The network is trained using repeated updates:

learning_rate = 0.1
epochs = 10000

The weights and biases are adjusted after each training step so that the predictions gradually become closer to the expected XOR outputs.

Results

After training for 10,000 epochs, my network produced:

Input| Expected| Predicted
[0, 0]| 0| 0.03
[0, 1]| 1| 0.96
[1, 0]| 1| 0.95
[1, 1]| 0| 0.06

Final Mean Squared Error:

0.0022921164199390244

The predictions are close to the expected XOR outputs, showing that the network successfully learned the pattern.

What I Learned

While making this project, I learned how the main parts of a neural network work together:

- How weights and biases affect predictions
- Why a hidden layer is useful for the XOR problem
- How the sigmoid activation function works
- How forward propagation produces predictions
- How backpropagation calculates updates
- How gradient descent improves the model
- How the loss value can be used to monitor training
- How NumPy can be used for basic neural-network calculations

The main purpose of this project was not to build a production-ready model, but to understand the mathematics and logic behind neural networks.

Technologies Used

- Python
- NumPy
- GitHub

Project Structure

neural-network-xor-backpropagation/
│
├── xor-neural-network.py
├── README.md
└── .gitignore

Requirements

Install NumPy:

pip install numpy

Then run:

python xor-neural-network.py

Possible Improvements

Some things I would like to add or experiment with in the future:

- Plot the loss during training
- Compare different learning rates
- Try different numbers of hidden neurons
- Experiment with different activation functions
- Add numerical gradient checking
- Test the network on other small datasets

Why I Made This Project

I made this project to get a better understanding of Artificial Intelligence and neural networks at a basic mathematical level.

Instead of directly using a machine-learning framework, I wanted to see what happens inside a neural network during training and understand the role of forward propagation, backpropagation, weights, biases, and loss.

Author

Manesh Kumar

Student interested in:

- Artificial Intelligence
- Computer Science
- Programming
- Machine Learning

---

This is a learning project created to understand the basic mathematics and implementation of a neural network.
Objective

The main objective of this project was to understand how a basic neural network works internally without depending on high-level machine-learning frameworks.

Specifically, the project demonstrates:

- Forward propagation
- Non-linear activation using the sigmoid function
- Error calculation
- Backpropagation
- The chain rule
- Gradient-based weight updates
- Learning a non-linearly separable function

Network Architecture

The network uses the following architecture:

2 Input Neurons
       ↓
4 Hidden Neurons
       ↓
1 Output Neuron

In mathematical form:

Input → Hidden Layer → Output
  2          4            1

The hidden layer allows the network to model the non-linear relationship required by XOR.

Mathematical Foundation

1. Sigmoid Activation

The sigmoid function is used as the activation function:

σ(z) = 1 / (1 + e^(-z))

Its derivative can be expressed using the sigmoid output:

σ'(z) = σ(z)(1 - σ(z))

In the implementation, the derivative is therefore calculated as:

def dsigmoid(x):
    return x * (1 - x)

where "x" is already the sigmoid output.

2. Forward Propagation

The hidden layer calculates a weighted combination of the inputs and applies the sigmoid function:

H = sigmoid(XW₁ + b₁)

The output layer then performs another weighted transformation:

Ŷ = sigmoid(HW₂ + b₂)

The result "Ŷ" represents the network's predicted output.

3. Error and Loss

The difference between the expected output and the prediction is calculated as:

error = y - ŷ

The project can use Mean Squared Error to measure the prediction error:

MSE = mean((y - ŷ)²)

4. Backpropagation

Backpropagation calculates how the error should influence the network parameters.

The output-layer error signal is calculated using the sigmoid derivative:

δ₂ = (y - ŷ) · σ'(ŷ)

The error is then propagated backward to the hidden layer:

δ₁ = (δ₂W₂ᵀ) · σ'(H)

These calculations apply the chain rule of differentiation to determine how changes in the weights affect the network's output.

5. Parameter Updates

The weights are updated iteratively using a learning rate:

W₂ ← W₂ + η(Hᵀδ₂)

W₁ ← W₁ + η(Xᵀδ₁)

where "η" represents the learning rate.

Implementation

The neural network is implemented using:

- Python 3
- NumPy

No high-level machine-learning frameworks such as TensorFlow, PyTorch, or scikit-learn are required for the core neural-network implementation.

The main operations are implemented manually using NumPy matrix multiplication and element-wise calculations.

Example Results

After training, the network produces outputs close to the expected XOR values.

Example:

Input| Predicted Output| Expected
"(0, 0)"| 0.08| 0
"(0, 1)"| 0.93| 1
"(1, 0)"| 0.92| 1
"(1, 1)"| 0.07| 0

The outputs are sigmoid values, so they are not required to be exactly "0" or "1". Values close to 0 represent the negative class, while values close to 1 represent the positive class.

«Note: Exact numerical results can vary depending on random initialization, network configuration, learning rate, and training settings.»

Training

The original implementation trains the network for:

10,000 epochs

with a learning rate of:

0.1

The weights are initialized using NumPy's random number generator.

For reproducibility, a fixed random seed can be used:

np.random.seed(1)

Why XOR?

XOR is useful for demonstrating the importance of non-linearity in neural networks.

The problem cannot be solved by simply drawing one straight decision boundary between the two classes.

The hidden layer and non-linear sigmoid activation allow the network to learn a more complex mapping.

This makes XOR a small but useful demonstration of the principles behind larger neural-network systems.

What I Learned

Through this project, I developed a practical understanding of:

- Matrix-based neural-network computation
- Forward propagation
- Activation functions
- Derivatives and the chain rule
- Backpropagation
- Gradient-based learning
- The importance of non-linear activation functions
- Why hidden layers are useful for non-linearly separable problems
- Implementing machine-learning concepts without relying on high-level frameworks

Future Improvements

Possible extensions to this project include:

- Adding bias parameters to every layer
- Recording and visualizing training loss
- Comparing different learning rates
- Experimenting with different hidden-layer sizes
- Comparing sigmoid with other activation functions
- Implementing binary cross-entropy loss
- Adding numerical gradient checking
- Testing the implementation on larger datasets

Project Structure

neural-network-xor/
│
├── README.md
├── xor_neural_network.py
├── requirements.txt
└── results/
    └── loss_curve.png

Requirements

Install NumPy with:

pip install numpy

Then run the Python implementation:

python xor_neural_network.py

Project Type

Area: Artificial Intelligence / Machine Learning
Mathematical Topics: Linear Algebra, Derivatives, Chain Rule, Optimization
Programming Language: Python
Library: NumPy
Level: Educational / Student Project

Author

Manesh Kumar

This project was developed as a hands-on exploration of the mathematical principles behind neural networks and machine learning.
