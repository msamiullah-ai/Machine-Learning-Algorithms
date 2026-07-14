# Neural Networks

Each neuron performs two operations.

## 1. Linear Combination

The neuron first computes a weighted sum of the inputs.

$$
z = W^T x + b
$$

where:

- x = input vector
- W = weight vector
- b = bias
- z = linear combination (also called the pre-activation value)

**Intuition:**

This step simply combines all input features according to their importance (weights) and shifts the result using the bias.

---

## 2. Activation Function

The output of the linear combination is then passed through an activation function.

$$
a = g(z)
$$

where **g(·)** can be any activation function, for example:

- Sigmoid
- ReLU
- Tanh

The activation converts the linear output into the neuron's final output.

### If the activation function is Sigmoid

$$
g(z)=\frac{1}{1+e^{-z}}
$$

The sigmoid function maps every real number into the interval

$$
(0,1)
$$

making it useful when the output represents a probability.

### If the activation function is ReLU

$$
g(z)=\max(0,z)
$$

This means:

- if **z < 0**, output = 0
- if **z ≥ 0**, output = z

ReLU is computationally efficient and is the most commonly used activation function in hidden layers of modern neural networks.

---

## Neuron + Sigmoid Activation = Logistic Regression Unit

When a neuron uses the sigmoid activation function, its output becomes

$$
h(x)=\sigma(W^Tx+b)
$$

where

- h(x) is the prediction made by the neuron.
- **σ(·)** denotes the sigmoid function.

This is exactly the mathematical model used in Logistic Regression.

---

## Why do we need an Activation Function?

The activation function introduces non-linearity into the network.

Without an activation function, every layer would perform only linear operations, and even a deep neural network would behave like a single linear model.

**Key Idea:**

Activation functions make neural networks capable of learning complex, non-linear relationships in data.

---

# Hidden Layer Representation

Suppose the output layer is given by

$$
h(x)=\sigma\left(w^T\phi(x)+v\right)
$$

where:

- w = weight vector connecting the hidden layer to the output neuron.
- v = bias of the output neuron.
- **σ(·)** = sigmoid activation function.
- φ(x) = feature representation (output of the hidden layer).

## What is φ(x)?

$$
\phi(x)
$$

is a learned feature vector, where each element represents the activation of one neuron in the hidden layer.

If the hidden layer computes

$$
z = Vx + c
$$

then after applying the activation function,

$$
\phi(x)=g(Vx+c)
$$

where

- V = weight matrix of the hidden layer
- c = bias vector
- **g(·)** = activation function (ReLU, Sigmoid, Tanh, etc.)

Thus,

φ(x) is simply the vector of activations coming out of the hidden layer.

Instead of using the original input directly, the output neuron makes its prediction using these learned features.

---

# Effect of Increasing the Number of Hidden Neurons

As the number of hidden neurons increases,

- the learned decision boundary becomes smoother,
- the network gains more flexibility,
- and it becomes capable of approximating increasingly complex functions.

Eventually, with a sufficient number of hidden neurons, a neural network can approximate almost any nonlinear function or decision boundary.

This idea is the foundation of the **Universal Approximation Theorem**.

**Intuition:**

More hidden neurons provide more building blocks that the network can combine to model complicated patterns in the data.

---

# Why Does This Work?

Although each individual neuron contributes only a simple piece of the final function (often behaving like a flat or straight, tilted segment after activation),

combining many such neurons allows the network to approximate a smooth and highly complex curve.

Think of it as drawing a complicated curve using many small line segments:

- One segment cannot represent the entire curve.
- Hundreds or thousands of segments together can closely match almost any shape.

---

# Neural Networks for Regression

A neural network performs regression by stitching together many simple linear pieces to approximate a complex nonlinear continuous function.

In other words,

> **Many simple neurons + nonlinear activation functions = a powerful model capable of learning highly complex continuous functions.**

---

# Why Do Neural Networks Prefer SGD?

Or, more precisely,

## Why Mini-Batch Stochastic Gradient Descent?

### Problem with Batch Gradient Descent (BGD)

Batch Gradient Descent computes the gradient using the entire training dataset before updating the weights.

To reliably find the global minimum, the optimization landscape should ideally be convex.

However,

The cost functions of neural networks are highly non-convex.

This means they contain many:

- Local minima
- Saddle points
- Flat regions

As a result, Batch Gradient Descent is not guaranteed to find the global optimum.

---

## Another Drawback of Batch Gradient Descent

For large neural networks,

Batch Gradient Descent becomes **computationally expensive and slow** because every parameter update requires processing the **entire dataset**.

When the dataset contains millions of examples, a single update can take a long time.

---

## Why Modern Deep Learning Uses Mini-Batch SGD

Instead of using the complete dataset,

modern neural networks typically use **Mini-Batch Stochastic Gradient Descent (Mini-Batch SGD).**

Mini-Batch SGD updates the network parameters after processing a **small batch of training examples** rather than waiting for the entire dataset.

This provides several advantages:

- Faster parameter updates
- Lower memory usage
- Better scalability
- Slight randomness in updates, which often helps escape saddle points and poor local minima

---

## Stochastic Gradient Descent (SGD)

SGD updates the model parameters

> **after every individual training observation (or data point).**

Mini-Batch SGD lies between Batch Gradient Descent and pure SGD:

- **Batch GD:** Update after the entire dataset.
- **SGD:** Update after every single training example.
- **Mini-Batch SGD:** Update after a small batch (e.g., 32, 64, 128 samples).

Because it balances computational efficiency and stable learning, **Mini-Batch SGD is the standard optimization method used in modern deep learning.**

---

# Matrix Dimensions in Forward Propagation

Consider the linear equation

$$
Z = W^T X + b
$$

## Dimensions

### Weight vector

$$
W \in \mathbb{R}^{(n,1)}
$$

Therefore,

$$
W^T \in \mathbb{R}^{(1,n)}
$$

### Input matrix

$$
X \in \mathbb{R}^{(n,m)}
$$

where

- n = number of input features
- m = number of training examples in the batch

### Bias

$$
b \in \mathbb{R}^{(1,1)}
$$

The bias is **broadcast** automatically across all **m** examples.

---

## Dimension Check

$$
(1,n)\times(n,m)
=
(1,m)
$$

Adding the bias does not change the dimensions.

Therefore,

$$
\boxed{Z\in\mathbb{R}^{(1,m)}}
$$

This means the neuron produces **one output value for each training example** in the mini-batch.

---

# Forward Propagation with One Training Example

Consider the following neural network:

```text
Input Layer (Layer 0)  →  Hidden Layer (Layer 1)  →  Output Layer (Layer 2)

a₁⁽⁰⁾
a₂⁽⁰⁾  ───────────────►  a₁⁽¹⁾
a₃⁽⁰⁾                   a₂⁽¹⁾ ───────────────► a₁⁽²⁾
a₄⁽⁰⁾                   a₃⁽¹⁾
```

Each neuron in the hidden layer computes its own linear combination independently.

For example,

$$
z_1^{[1]}=(w_1^{[1]})^Ta^{[0]}+b_1^{[1]}
$$

Similarly,

$$
z_2^{[1]}=(w_2^{[1]})^Ta^{[0]}+b_2^{[1]}
$$

Likewise, every hidden neuron computes

$$
z_i^{[1]}=(w_i^{[1]})^Ta^{[0]}+b_i^{[1]}.
$$

Instead of writing one equation for every neuron separately, we combine them into a **single vector equation**.

---

## Vector Form

Collecting the outputs of all hidden neurons,

$$
z^{[1]}=
\begin{bmatrix}
z_1^{[1]}\\
z_2^{[1]}\\
\vdots\\
z_n^{[1]}
\end{bmatrix}
$$

where

- each element corresponds to one neuron in Layer 1,
- all values are computed from the same input example.

This is the **pre-activation vector** for the hidden layer.

---

# Extension to m Training Examples

Instead of processing only **one training example**, neural networks usually process an entire **mini-batch** simultaneously.

Suppose we have

$$
m
$$

training examples.

Rather than storing a single vector,

we arrange all pre-activation vectors into a matrix.

---


## Important Interpretation

### 1. Each **row** represents one neuron

Every row tracks the output of **one hidden neuron** across **all training examples**.

For example,

- Row 1 contains the outputs of neuron 1 for every example.
- Row 2 contains the outputs of neuron 2 for every example.
- And so on.

---

### 2. Each **column** represents one training example

Every column contains the outputs of **all neurons** for a **single training example**.

In other words,

- Column 1 corresponds to Training Example 1.
- Column 2 corresponds to Training Example 2.
- ...
- Column **m** corresponds to Training Example **m**.

---

# Matrix Representation

The pre-activation matrix for Layer 1 becomes

$$
Z^{[1]}=
\begin{bmatrix}
z_1^{[1](1)} & z_1^{[1](2)} & \cdots & z_1^{[1](m)} \\
z_2^{[1](1)} & z_2^{[1](2)} & \cdots & z_2^{[1](m)} \\
\vdots & \vdots & \ddots & \vdots \\
z_n^{[1](1)} & z_n^{[1](2)} & \cdots & z_n^{[1](m)}
\end{bmatrix}
$$

where

- superscript **[1]** denotes **Layer 1**,
- superscript **(i)** denotes the **iᵗʰ training example**.

Thus,

- **Rows = neurons**
- **Columns = training examples**

This organization is the standard representation used in vectorized neural network implementations.

---

# Vectorized Forward Propagation for a Mini-Batch

Suppose we have **m training examples**.

The pre-activation values of the **first neuron** for all training examples can be written as

$$
Z_1^{[1]}=
\begin{bmatrix}
z_1^{(1)} & z_1^{(2)} & z_1^{(3)} & \cdots & z_1^{(m)}
\end{bmatrix}
$$

where

- each entry is the pre-activation of **Neuron 1**,
- each value corresponds to **one training example**.

So,

> **One row stores the pre-activation values of one neuron across the entire mini-batch.**

---

# Combined Pre-Activation Matrix for an Entire Layer

Instead of writing the activations of each neuron separately, we stack them together into a single matrix.

$$
Z=
\begin{bmatrix}
Z_1 \\
Z_2 \\
Z_3 \\
\vdots \\
Z_n
\end{bmatrix}
$$

or equivalently,

$$
Z=
\begin{bmatrix}
z_1^{(1)} & z_1^{(2)} & \cdots & z_1^{(m)} \\
z_2^{(1)} & z_2^{(2)} & \cdots & z_2^{(m)} \\
z_3^{(1)} & z_3^{(2)} & \cdots & z_3^{(m)} \\
\vdots & \vdots & \ddots & \vdots \\
z_n^{(1)} & z_n^{(2)} & \cdots & z_n^{(m)}
\end{bmatrix}
$$

This matrix contains the pre-activation values of **every neuron** for **every training example**.

---

# Constructing the Weight Matrix

The weight matrix is organized in the same way.

Each **row** corresponds to **one neuron** in the current layer.

For a layer with **n** neurons and **d** input features,

$$
W=
\begin{bmatrix}
(w_1)^T \\
(w_2)^T \\
(w_3)^T \\
\vdots \\
(w_n)^T
\end{bmatrix}
$$

where

- **Row 1** contains all weights of **Neuron 1**.
- **Row 2** contains all weights of **Neuron 2**.
- **Row 3** contains all weights of **Neuron 3**.
- ⋮
- **Row n** contains all weights of **Neuron n**.

In other words,

$$
W_i=
\begin{bmatrix}
w_{i1} & w_{i2} & \cdots & w_{id}
\end{bmatrix}
$$

is the complete weight vector of the **iᵗʰ** neuron.

---

## Key Observation

The organization of the matrices is intentional:

- **Rows represent neurons.**
- **Columns represent training examples.**

This consistent arrangement allows the forward propagation of an entire mini-batch to be computed using a **single matrix multiplication**, rather than evaluating each neuron and each training example individually.

---

# Matrix of Activations

For Layer **l**, the activation matrix is

$$
A^{[l]}=
\begin{bmatrix}
a_1^{[l](1)} & a_1^{[l](2)} & \cdots & a_1^{[l](m)} \\
a_2^{[l](1)} & a_2^{[l](2)} & \cdots & a_2^{[l](m)} \\
a_3^{[l](1)} & a_3^{[l](2)} & \cdots & a_3^{[l](m)} \\
\vdots & \vdots & \ddots & \vdots \\
a_n^{[l](1)} & a_n^{[l](2)} & \cdots & a_n^{[l](m)}
\end{bmatrix}
$$

where:

- Each **row** represents the activations of one neuron.
- Each **column** represents one training example.
- **m** is the number of training examples in the mini-batch.

> **Interpretation:**  
> The activation matrix stores the outputs of **all neurons** for **all training examples** in a particular layer.

---

# Backpropagation

Backpropagation is the algorithm used to compute the gradients of the loss function with respect to every parameter (weights and biases) in the neural network.

It applies the **Chain Rule of Calculus** to efficiently propagate errors from the output layer back toward the input layer.

---

## Forward Pass

For a single neuron,

### Linear Combination

$$
z=w_1x_1+w_2x_2+b
$$

### Activation

$$
a=\sigma(z)
$$

### Loss Function

For binary classification,

$$
L(a,y)=-y\log(a)-(1-y)\log(1-a)
$$

where

- **a** = predicted probability
- **y** = true label

The computational flow is

$$
z \longrightarrow a \longrightarrow L(a,y)
$$

---

# Step 1: Compute the Gradient with Respect to the Activation

We first differentiate the loss with respect to the neuron's output.

$$
da=\frac{\partial L(a,y)}{\partial a}
$$

Differentiating the binary cross-entropy loss gives

$$
\boxed{\frac{\partial L}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}}
$$

This tells us **how sensitive the loss is to changes in the activation a.**

---

# Step 2: Compute the Gradient with Respect to z

Since

$$
L
$$

depends on

$$
z
$$

through

$$
a,
$$

we apply the **Chain Rule**.

$$
\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\times\frac{\partial a}{\partial z}
$$

or equivalently,

$$
\boxed{dz=da\cdot\frac{da}{dz}}
$$

The dependency chain is

$$
\boxed{z\longrightarrow a\longrightarrow L}
$$

which is exactly why the Chain Rule is required.

> **Key Idea:**  
> During backpropagation, gradients are propagated **backwards** through the computational graph by repeatedly applying the Chain Rule.

---

# Backpropagation Derivation (Continued)

## Step 2: Compute dz

From the Chain Rule,

$$
dz=\frac{\partial L}{\partial z}=\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}
$$

From the previous derivation,

$$
\frac{\partial L}{\partial a}=-\frac{y}{a}+\frac{1-y}{1-a}
$$

Since

$$
a=\sigma(z),
$$

the derivative of the sigmoid function is

$$
\frac{da}{dz}=\sigma(z)(1-\sigma(z))=a(1-a)
$$

Substituting both results,

$$
dz=\left(-\frac{y}{a}+\frac{1-y}{1-a}\right)a(1-a)
$$

---

## Simplification

Expand the multiplication:

$$
dz=-y(1-a)+a(1-y)
$$

Now simplify:

$$
\begin{aligned}
dz &= -y+ay+a-ay \\
   &= a-y
\end{aligned}
$$

Hence,

$$
\boxed{dz=a-y}
$$

This is one of the most important equations in neural network backpropagation.

> **Key Result:**  
> For a sigmoid output neuron with binary cross-entropy loss,
>
> $$
> \boxed{dz=a-y}
> $$
>
> This elegant simplification is one reason why the combination of **sigmoid activation** and **binary cross-entropy loss** is widely used.

---

# Gradient with Respect to the Weight wᵢ

Next, we compute how the loss changes with respect to a weight.

The computational dependency is

$$
w_i\longrightarrow z\longrightarrow a\longrightarrow L
$$

Applying the Chain Rule,

$$
\frac{\partial L}{\partial w_i}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w_i}
$$

Since

$$
\frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}=dz=a-y,
$$

we obtain

$$
\frac{\partial L}{\partial w_i}=(a-y)\frac{\partial z}{\partial w_i}
$$

---

## Derivative of z with Respect to wᵢ

Recall the linear equation

$$
z=w_1x_1+w_2x_2+\cdots+w_ix_i+b
$$

Differentiating with respect to **wᵢ**,

$$
\frac{\partial z}{\partial w_i}=x_i
$$

Substituting,

$$
\boxed{dw_i=(a-y)x_i}
$$

Since

$$
dz=a-y,
$$

this can also be written as

$$
\boxed{dw_i=dz\cdot x_i}
$$

which is the standard backpropagation equation for the gradient of a weight.

---

# Vectorized Backpropagation

From the single training example derivation, we obtained

$$
dz = a - y
$$

Now we extend this result to a **mini-batch** containing **m** training examples.

---

## Vectorized Gradient of Z

The activation matrix of the output layer is

$$
A \in \mathbb{R}^{(1,m)}
$$

where each column corresponds to one training example.

Similarly, the ground truth labels are

$$
Y \in \mathbb{R}^{(1,m)}.
$$

Therefore,

$$
\boxed{
dZ = A - Y
}
$$

### Dimension Check

Since

$$
A \in \mathbb{R}^{(1,m)}
$$

and

$$
Y \in \mathbb{R}^{(1,m)},
$$

we have

$$
dZ
=
A-Y
=
(1,m)-(1,m)
=
(1,m).
$$

Thus,

$$
\boxed{
dZ \in \mathbb{R}^{(1,m)}
}
$$

---

# Bias Gradient

For one training example,

the bias gradient is simply

$$
db = dz.
$$

For a mini-batch, we compute the **average error across all training examples**.

Hence,

$$
\boxed{
db
=
\frac{1}{m}
\sum_{i=1}^{m}
dz^{(i)}
}
$$

Since all individual errors are stored inside the vector

$$
dZ
=
\begin{bmatrix}
dz^{(1)} &
dz^{(2)} &
\cdots &
dz^{(m)}
\end{bmatrix},
$$

the above equation can be written compactly as

$$
\boxed{
db
=
\frac{1}{m}
\operatorname{Sum}(dZ)
}
$$

where the **Sum** operation adds all entries of **dZ**.

> **Interpretation:**  
> The bias gradient is simply the **average prediction error** over the entire mini-batch.

---

# Weight Gradient

For a single training example, we derived

$$
dw_i = dz \cdot x_i.
$$

To extend this to all training examples simultaneously, we vectorize the computation.

The gradient of the weight matrix is

$$
\boxed{
dW
=
\frac{1}{m}
X\,dZ^{T}
}
$$

---

## Dimension Check

Input matrix:

$$
X
\in
\mathbb{R}^{(n,m)}
$$

Gradient matrix:

$$
dZ
\in
\mathbb{R}^{(1,m)}
$$

Therefore,

$$
dZ^{T}
\in
\mathbb{R}^{(m,1)}.
$$

Now,

$$
(n,m)\times(m,1)
=
(n,1).
$$

Hence,

$$
\boxed{
dW
\in
\mathbb{R}^{(n,1)}
}
$$

which is exactly the same shape as the original weight vector.

---

## Summary of Vectorized Gradients

For logistic regression (or a single output neuron), the complete vectorized backpropagation equations are

$$
\boxed{
dZ=A-Y
}
$$

$$
\boxed{
dW=\frac1mXdZ^T
}
$$

$$
\boxed{
db=\frac1m\operatorname{Sum}(dZ)
}
$$

These equations compute the gradients for an **entire mini-batch** in one set of matrix operations, making neural network training highly efficient.

---

# Backpropagation for Hidden Layers

In the previous sections, we derived the gradients for the **output layer**.

Now we generalize the derivation for **any hidden layer** **l**.

---

## Forward Propagation Recap

For layer **l**,

### Linear Step

$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
$$

### Activation Step

$$
A^{[l]} = g\!\left(Z^{[l]}\right)
$$

where

- A^[l−1] = activations from the previous layer,
- W^[l] = weight matrix,
- b^[l] = bias vector,
- Z^[l] = pre-activation values,
- A^[l] = activations of the current layer.

---

# Step 1: Gradient of the Activation

Using the Chain Rule,

$$
\frac{\partial L}{\partial A^{[l]}}
=
dA^{[l]}
$$

This quantity represents how sensitive the loss is to the activations of layer **l**.

---

# Step 2: Gradient of the Linear Output

Since

$$
A^{[l]} = g\!\left(Z^{[l]}\right),
$$

the Chain Rule gives

$$
\frac{\partial L}{\partial Z^{[l]}}
=
\frac{\partial L}{\partial A^{[l]}}
\odot
g'\!\left(Z^{[l]}\right),
$$

where

- **⊙** denotes **element-wise multiplication** (Hadamard product),
- **g′(Z)** is the derivative of the activation function.

Therefore,

$$
\boxed{
dZ^{[l]}
=
dA^{[l]}
\odot
g'\!\left(Z^{[l]}\right)
}
$$

---

# Step 3: Gradient of the Weight Matrix

Recall

$$
Z^{[l]}
=
W^{[l]}A^{[l-1]}
+
b^{[l]}.
$$

Differentiating with respect to the weights gives

$$
\boxed{
dW^{[l]}
=
\frac1m
\,dZ^{[l]}
\left(A^{[l-1]}\right)^T
}
$$

--- 

## Dimension Check

Suppose

- dZ^[l] ∈ ℝ^(nₗ,m)
- A^[l−1] ∈ ℝ^(nₗ₋₁,m)

Then

$$
(n_l,m)
\times
(m,n_{l-1})
=
(n_l,n_{l-1}),
$$

which matches the dimensions of

$$
W^{[l]}.
$$

---

# Step 4: Gradient of the Bias

The bias is shared across all training examples.

Hence,

$$
\boxed{
db^{[l]}
=
\frac1m
\sum dZ^{[l]}
}
$$

where the summation is performed **column-wise** (over the mini-batch).

---

# Step 5: Gradient Passed to the Previous Layer

Finally, we compute the gradient required for the previous layer.

Applying the Chain Rule,

$$
\boxed{
dA^{[l-1]}
=
\left(W^{[l]}\right)^T
dZ^{[l]}
}
$$

## Dimension Check

If

- W^[l] ∈ ℝ^(nₗ,nₗ₋₁),

then

$$
\left(W^{[l]}\right)^T
\in
\mathbb{R}^{(n_{l-1},n_l)}.
$$

Therefore,

$$
(n_{l-1},n_l)
\times
(n_l,m)
=
(n_{l-1},m),
$$

which is exactly the shape of

$$
A^{[l-1]}.
$$

---

# Complete Backpropagation Equations

For every hidden layer **l**,

$$
\boxed{
dZ^{[l]}
=
dA^{[l]}
\odot
g'\!\left(Z^{[l]}\right)
}
$$

$$
\boxed{
dW^{[l]}
=
\frac1m
dZ^{[l]}
\left(A^{[l-1]}\right)^T
}
$$

$$
\boxed{
db^{[l]}
=
\frac1m
\sum dZ^{[l]}
}
$$

$$
\boxed{
dA^{[l-1]}
=
\left(W^{[l]}\right)^T
dZ^{[l]}
}
$$

These four equations form the core of the backpropagation algorithm used to train deep neural networks.
