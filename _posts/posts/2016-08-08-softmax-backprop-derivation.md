---
layout: post
title: "Backpropagation with softmax outputs and cross-entropy cost"
description: "Deriving the backpropagation algorithm for a fully-connected multi-layer neural network with softmax output layer and log-likelihood cost function."
keywords: "backpropagation, neural networks, deep learning, machine learning, softmax, log-likelihood"
visible: 1
---

In a [previous post](/2016/gd-backprop-derivation) we derived the 4 central equations of backpropagation in full generality, while making very mild assumptions about the cost and activation functions. In this post, we'll derive the equations for a concrete cost and activation functions. 

We are going to re-use the [notation](/2016/gd-backprop-derivation/#backpropagation) that was formulated in the previous post. I suggest you take a quick look at this, just to make sure you are comfortable with these pieces of notations.<br><br>  

### Network architecture
We are going to work with a fully-connected neural network, which has **sigmoid** activation functions in its hidden layers and **softmax** activation functions in its outer layer. The performance of our network is measured by the **cross-entropy** cost function.

We also assume that the desired output of the network for a single training example \\(x\\) is given by a **one-hot vector** \\(y\\), whose \\(i^{th}\\) element is \\(1\\) \\((y_i = 1)\\) and whose all other elements are \\(0\\) \\((y_{j \neq i} = 0)\\). This is a natural assumption to make for classification tasks.<br><br>

### Sigmoid hidden layer
In the sigmoid hidden layer, the activation \\(a_{i}^{l}\\) of the \\(i^{th}\\) neuron is given by the sigmoid function: \\[a_i^l = \frac{1}{1 + e^{-z_i^l}}. \\]

Later, when deriving the backpropagation equations for our network, we'll need to know the **rate of change of the sigmoid unit's activation w.r.t. its input:** \\[\begin{equation} \frac{d a_i^l}{d z_i^l} = a_i^l (1 - a_i^l), \end{equation} \label{eq:delta_sigmoid}\\] where we have used the regular derivative instead of the partial derivative because \\(z^l\\) is the only variable \\(a^l\\) depends on.

**Proof:** \\[\frac{d a_i^l}{d z_i^l} \stackrel{a \text{ def}}{=} \frac{d}{d z_i^l} \left( \frac{1}{1 + e^{-z_i^l}} \right) = \frac{e^{-z_i^l}}{(1 + e^{-z_i^l}) ^ 2} = \frac{1}{1 + e^{-z_i^l}} \cdot \frac{e^{-z_i^l}}{1 + e^{-z_i^l}} \stackrel{+- 1}{=} \\\ \frac{1}{1 + e^{-z_i^l}} \cdot \frac{1 + e^{-z_i^l} - 1}{1 + e^{-z_i^l}} = \frac{1}{1 + e^{-z_i^l}} \left( 1 - \frac{1}{1 + e^{-z_i^l}} \right) \stackrel{a \text{ def}}{=} a_i^l (1 - a_i^l). \quad \blacksquare \\]<br>

### Softmax output layer
In the softmax output layer, the activation \\(a_{i}^{L}\\) of the \\(i^{th}\\) neuron is given by the softmax function: \\[\begin{equation} a_{i}^{L} = \frac{e^{z_{i}^{L}}}{\sum_j e^{z_{j}^{L}}}. \end{equation} \label{eq:softmax}\\]

There are two interesting properties to note about the softmax activation function \\(\eqref{eq:softmax}\\). First, the output of the softmax layer can be thought of as a probability distribution, since all the activations of the output layer sum up to 1: \\[\sum_i a_{i}^{L} \stackrel{a \text{ def}}{=} \sum_i \frac{e^{z_{i}^{L}}}{\sum_j e^{z_{j}^{L}}} = \frac{\sum_i e^{z_{i}^{L}}}{\sum_j e^{z_{j}^{L}}} = 1.\\]  

Secondly, the input \\(z_{i}^{L}\\) not only influences activation \\(a_{i}^{L}\\), but **all** the other activations \\(a_j^{L}\\) as well, through the regularization term in the denominator of \\(\eqref{eq:softmax}\\). This invalidates the assumption made in the previous post about the activation function (i.e. the input \\(z_{i}^{L}\\) should influence activation \\(a_{i}^{L}\\) **only**), yielding the [first equation of backpropagation](/2016/gd-backprop-derivation/#bp1) (the equation for the error in the output layer) derived there inapplicable for the softmax function.

Later, when deriving the backpropagation equations for our network, we'll need two relations, which we'll formulate next.

**1. Rate of change of the softmax unit's activation w.r.t. its input:** \\[\begin{equation} \frac{\partial a_i^L}{\partial z_i^L} = a_i^L (1 - a_i^L). \end{equation} \label{eq:delta_softmax_i}\\]

**Proof:** \\[\frac{\partial a_i^L}{\partial z_i^L} \stackrel{a \text{ def}}{=} \frac{\partial}{\partial z_i^L} \left(\frac{e^{z_i^L}}{\sum_j e^{z_j^L}} \right) \stackrel{\text{quotient rule}}{=} \frac{\left( \sum_j e^{z_j^L} \right) e^{z_i^L} - \left( e^{z_i^L} \right)^2}{\left( \sum_j e^{z_j^L} \right) ^ 2} = \\\ \frac{e^{z_i^L} \left( \sum_j e^{z_j^L} - e^{z_i^L} \right)}{\left( \sum_j e^{z_j^L} \right) ^ 2} = \frac{e^{z_i^L}}{\sum_j e^{z_j^L}} \cdot \left( \frac{\sum_j e^{z_j^L}}{\sum_j e^{z_j^L}} - \frac{e^{z_i^L}}{\sum_j e^{z_j^L}} \right) \stackrel{a \text{ def}}{=} a_i^L (1 - a_i^L). \quad \blacksquare\\]

**2. Rate of change of the softmax unit's activation w.r.t. the input to some other softmax unit:** \\[\begin{equation} \frac{\partial a_j^L}{\partial z_i^L} = -a_j^L a_i^L, \end{equation} \label{eq:delta_softmax_j} \\] where \\(j \neq i\\).

**Proof:** \\[\frac{\partial a_j^L}{\partial z_i^L} \stackrel{a \text{ def}}{=} \frac{\partial}{\partial z_i^L} \left( \frac{e^{z_j^L}}{\sum_k e^{z_k^L}} \right) \stackrel{\text{quotient rule}}{=} \frac{-e^{z_j^L} e^{z_i^L}}{\left( \sum_k e^{z_k^L} \right) ^ 2} = \\\ -\frac{e^{z_j^L}}{\sum_k e^{z_k^L}} \cdot \frac{e^{z_i^L}}{\sum_k e^{z_k^L}}  \stackrel{a \text{ def}}{=} -a_j^L a_i^L. \quad \blacksquare\\]<br>

### Cross-entropy cost function
The cross-entropy cost is given by \\[C = -\frac{1}{n} \sum_x \sum_i y_i \ln a_{i}^{L},\\] where the inner sum is over all the softmax units in the output layer.

For a single training example, the cost becomes \\[C_x = -\sum_i y_i \ln a_{i}^{L}.\\]

Note that since our target vector \\(y\\) is one-hot (a realistic assumption that we made earlier), the equation for the cross-entropy cost function further simplifies to \\[C_x = -\ln a_j^L,\\] where \\(j\\) is the index of the element in the vector \\(y\\) that is "hot" (i.e. equal to \\(1\\)) [[^1]].<br><br>

### Deriving the equations of backpropagation
We have now developed all the necessary tools to tackle the 4 equations of backpropagation for the proposed network architecture.  

**1. Error in the output layer:** \\[\delta_i^L = a_i^L - y_i.\\]

**Matrix form:** \\[\delta^L = a^L - y.\\]

**Proof:** \\[\delta_i^L \stackrel{\delta \text{ def}}{=} \frac{\partial C_x}{\partial z_i^L} \stackrel{C_x \text{ def}}{=} \frac{\partial}{\partial z_i^L} \left( - \sum_j y_j \ln a_j^L \right) = -\sum_j y_j \frac{\partial \ln a_j^L}{\partial z_i^L} = - \sum_j \frac{y_j}{a_j^L} \cdot \frac{\partial a_j^L}{\partial z_i^L}.\\]

Since the activation \\(a_j^L\\) depends on all \\(z_j^L\\) and its derivative is different for \\(z_i^L\\) and \\(z_j^L\\) \\((j \neq i)\\), we must split the sum to deal with the 2 cases separately: \\[\delta_i^L = -\frac{y_i}{a_i^L} \cdot \frac{\partial a_i^L}{\partial z_i^L} - \sum_{j \neq i} \frac{y_j}{a_j^L} \cdot \frac{\partial a_j^L}{\partial z_i^L}.\\]

We can now apply the 2 partial derivatives \\(\eqref{eq:delta_softmax_i}\\) and \\(\eqref{eq:delta_softmax_j}\\) derived earlier: \\[\delta_i^L = -\frac{y_i}{a_i^L} a_i^L(1 - a_i^L) + \sum_{j \neq i} \frac{y_j}{a_j^L} a_j^L a_i^L = -y_i (1 - a_i^L) + \sum_{j \neq i} y_j a_i^L = \\\ -y_i + a_i^L y_i + a_i^L \sum_{j \neq i} y_j = a_i^L \left( y_i + \sum_{j \neq i } y_j\right) - y_i = a_i^L \left( \sum_j y_j \right) - y_i.\\]

Since the vector \\(y\\) is one-hot, \\(\sum_j y_j = 1\\) and the equation simplifies to \\[\delta_i^L = a_i^L - y_i. \quad \blacksquare\\] 

**The remaning 3 equations ([BP2](/2016/gd-backprop-derivation/#bp2), [BP3](/2016/gd-backprop-derivation/#bp3),[BP4](/2016/gd-backprop-derivation/#bp4))** remain identical to the ones derived in the previous article, since the sigmoid cost function satisfies the properties we assumed (i.e. \\(z_i^l\\) influences \\(a_i^l\\) only). We just need to plug the relations derived in this chapter into the general equations (e.g. relation \\(\eqref{eq:delta_sigmoid}\\) into BP2 as \\(\delta_j^{l+1}\\)).<br><br>

---
**Footnote:**

[^1]: 1: This simplified form of cross-entropy is sometimes also referred to as the log-likelihood cost function.  