---
layout: post
title: "Gradient descent and backpropagation"
description: "Deriving the backpropagation algorithm for a fully-connected multi-layer neural network."
keywords: "backpropagation, neural networks, gradient descent, deep learning, machine learning"
visible: 1
---

*(This post is nothing more than a concise summary of the first 2 chapters of Michael Nielsen's excellent online book "Neural Networks and Deep Learning" [[^1]]. When implementing neural networks from scratch, I constantly find myself referring back to these
2 chapters for details on gradient descent and backpropagation. Thus, I thought it would be practical to have the relevant pieces of information laid out here in a more compact form for quick reference.)*<br><br> 

## Gradient descent 
Gradient descent is an iterative optimization algorithm that is used to find the local minimum of a function. It is iterative in the sense that the algorithm repeatedly takes small steps towards the minimum until it converges.

It is important to note that gradient descent is not specific to neural networks - the algorithm can be used to optimize **any** differentiable function with any number of arguments. Thus, in the following analysis, we concentrate on an arbitrary function of 2 variables, and later make the connection to neural networks explicit. 

Suppose we are trying to minimize \\(C(v)\\), which is a function of 2 variables \\(v=v\_1,v\_2\\). By changing the variables \\(v\_1\\) and \\(v\_2\\) by a small amount \\(\Delta v\_1\\) and \\(\Delta v\_2\\), respectively, the change in the function's value is approximated by: \\[\Delta C \approx \frac{\partial C} {\partial v\_1} \Delta v\_1 + \frac{\partial C} {\partial v\_2} \Delta v\_2.\\]

This relation can be written more compactly in a vectorized form using the dot product: \\[\begin{equation} \Delta C \approx \nabla C \cdot \Delta v, \end{equation} \label{eq:delta_C}\\] where \\[\nabla C = \left(\frac{\partial C} {\partial v\_1}, \frac{\partial C} {\partial v\_2}\right)^T \quad \text{and} \quad \Delta v = \left( \Delta v\_1, \Delta v\_2 \right)^T.\\]

Now, since we want to minimize \\(C\\), we would like to change the parameter \\(v\\) in a way that makes \\(\Delta C\\) negative. This can be easily done by choosing \\[\begin{equation} \Delta v = -\eta \nabla C,\end{equation} \label{eq:delta\_v}\\] where \\(\eta > 0\\) is the learning rate. Indeed, by substituting \\(\Delta v\\) into equation \\(\eqref{eq:delta_C}\\), we see that \\(\Delta C\\) becomes negative: \\(\Delta C \approx -\eta \nabla C \cdot \nabla C = -\eta \lVert \nabla C \lVert ^ 2 \\) < 0.

By repeatedly calculating \\(\Delta v\\) using equation \\(\eqref{eq:delta\_v}\\) and updating the parameter \\(v\\) accordingly, we will eventually reach a local minimum of \\(C\\). The update rule for \\(v\\) is given by the following relation:
\\[\begin{equation} v\_{new} = v\_{old} + \Delta v\_{old} = v\_{old} - \eta \nabla C. \end{equation} \label{eq:update_v}\\]

Finally, we have acquired the tools to precisely define the gradient descent algorithm, and as it turns out, the definition is surprisinly simple. **The gradient descent algorithm is nothing more than just iteratively applying the update rule \\(\eqref{eq:update_v}\\) until convergence.**<br><br>

#### Gradient descent in neural networks
Now that we know how to derive gradient descent for an arbitrary multivariate function, we can explicitly state it in the context of artificial neural networks, and analyse some of the problems that might arise due to this specific context.

In neural networks, we are trying to minimize a cost function \\(C(w, b)\\) w.r.t. weights \\(w\\) and biases \\(b\\). The cost function \\(C\\) can usually be written as an average over costs \\(C_{x_i}\\) of \\(n\\) individual training examples \\(x_i \quad (i=1,...,n)\\): \\[C = \frac{1}{n} \sum_{i=1}^{n} C_{x_i}.\\]

In order to apply the update rule \\(\eqref{eq:update_v}\\), we would have to calculate \\(\nabla C = \frac{1}{n} \sum_{i=1}^{n} \nabla C_{x_i}\\). That is, we would have to compute \\(\nabla C_{x_i}\\) for **every** training example in our training set to make a **single** update to parameters \\(w\\) and \\(b\\). That is a problem, because we could have millions of training examples in our dataset, yielding an extremely slow convergence of the gradient descent optimization algorithm.

To mitigate this problem, we could estimate the exact gradient \\(\nabla C\\) by computing \\(\nabla C_{x_i}\\) for \\(m \ll n\\) random training examples and averaging over this smaller sample: \\[\begin{equation}  \frac{1}{m} \sum_{i=1}^{m} \nabla C_{x_i} \approx \frac{1}{n} \sum_{i=1}^{n} \nabla C_{x_i} = \nabla C. \end{equation} \label{eq:batch_gd} \\] 

This idea is called **stochastic gradient descent**, and it usually quarantees much faster convergence at the expense of noisier gradient (which might not necessarily be a bad thing, because it could jerk the model out of local minima).<br><br>

## Backpropagation
Backpropagation is a method that **efficiently** calculates the gradient of the loss function w.r.t. all the weights and biases in the network. This gradient can then be fed into the gradient descent update rule \\(\eqref{eq:update_v}\\) to update the parameters of the network. Backpropagation is an efficient algorithm in the sense that it only needs 2 passes through the whole network (a forward and a backward pass) to compute the gradient \\(\nabla C_x\\) of a single training example \\(x\\).

We are going to derive the equations that allow us to calculate the gradient \\(\nabla C_x\\). In order to do so, we first need to agree on some relevant notation [[^2]]:

* \\(w_{ij}^{l}\\) - the weight for the connection from \\(i^{th}\\) neuron in the \\((l-1)^{th}\\) layer to the \\(j^{th}\\) neuron in the \\(l^{th}\\) layer;
* \\(b_{i}^{l}\\) - the bias for the \\(i^{th}\\) neuron in the \\(l^{th}\\) layer;
* \\(z_{i}^{l}\\) - the input to the \\(i^{th}\\) neuron in the \\(l^{th}\\) layer;
* \\(a_{i}^{l}\\) - the activation of the \\(i^{th}\\) neuron in the \\(l^{th}\\) layer;
* \\(\delta_{i}^{l} = \frac{\partial C_x}{\partial z_{i}^{l}}\\) - the error of neuron \\(i\\) in layer \\(l\\). This quantity is called error because the larger its value, the more the cost decreases by changing the input \\(z_{i}^{l}\\).

The activation of the \\(i^{th}\\) neuron in the \\(l^{th}\\) layer is related to the activations of the neurons in the \\((l-1)^{th}\\) layer: \\[a_{i}^{l} = a(\sum_{j} w_{ji}^{l} a_{j}^{l-1} + b_{i}^{l}),\\] where the argument inside the parenthesis is \\(z_{i}^{l}\\). Note that the only assumptions we make about the activation function \\(a_{i}^{l}\\) is that it is differentiable and that \\(z_{i}^{l}\\) influences the value of \\(a_i^l\\) **only** (as opposed to influencing \\(a_j^l\\) as well, where \\(j \neq i\\)) [[^3]][[^4]]. Not concentrating on a specific form of the activation function allows us to derive the equations of backpropagation in full generality. 

Armed with these pieces of notation, we are ready to derive the equations of backpropagation. Michael Nielsen divides the backpropagation algorithm into 4 fundamental equations [[^5]], and I'm going to follow the same convention in this article. I define each equation in both component and matrix forms and give a proof of the equation in component form.

<a name="bp1"></a>**1. Error in the output layer:** \\[\begin{equation} \tag{BP1} \delta_{i}^{L} = \frac{\partial C_x}{\partial a_{i}^{L}} a'(z_{i}^{L}), \end{equation} \label{eq:backprop1}\\] where \\(L\\) is the index of the output layer of the network.

**Matrix form:** \\[\delta^L = \nabla_a C_x \odot a' (z^L),\\] where \\(\odot\\) is the Hadamard (element-wise) product.

**Proof:** \\[\delta_{i}^{L} \stackrel{\text{def}}{=} \frac{\partial C_x}{\partial z_{i}^{L}} \stackrel{\text{chain rule}}{=} \sum_j \frac{\partial C_x}{\partial a_{j}^{L}} \cdot \frac{\partial a_{j}^{L}}{\partial z_{i}^{L}},\\] where the sum is over all neurons in the output layer. Because the activation \\(a_{j}^{L}\\) only depends on \\(z_{i}^{L}\\) if \\(j=i\\), all the other terms (where \\(j \neq i\\)) in the sum become \\(0\\) when we differentiate and the relation simplifies to \\[\delta_{i}^{L} = \frac{\partial C_x}{\partial a_{i}^{L}} \cdot \frac{\partial a_{i}^{L}}{\partial z_{i}^{L}} \stackrel{a \text{ def}}{=} \frac{\partial C_x}{\partial a_{i}^{L}} a'(z_{i}^{L}). \quad \blacksquare \\]

<a name="bp2"></a>**2. Error \\(\bf \delta^{l}\\) in terms of the error \\(\bf \delta^{l+1}\\) in the next layer:** \\[\begin{equation} \tag{BP2} \delta_{i}^{l} = \sum_{j} w_{ij}^{l+1} \delta_{j}^{l+1} a'(z_{i}^{l}). \end{equation} \label{eq:backprop2} \\]

**Matrix form:** \\[\delta^l = w^{l+1} \delta^{l+1} \odot a'(z^l),\\] where \\(w^{l+1}\\) is a weight matrix where the elements in row \\(i\\) represent the weights from neurons in layer \\(l\\) to the \\(i^{th}\\) neuron in layer \\(l+1\\).

**Proof:** \\[\delta_{i}^{l} \stackrel{\text{def}}{=} \frac{\partial C_x}{\partial z_{i}^{l}} \stackrel{\text{chain rule}}{=} \sum_{j} \frac{\partial C_x}{\partial z_{j}^{l+1}} \cdot \frac{\partial z_{j}^{l+1}}{\partial z_{i}^{l}} \stackrel{\delta \text{ def}}{=} \sum_{j} \delta_{j}^{l+1} \frac{\partial z_{j}^{l+1}}{\partial z_{i}^{l}} \stackrel{z \text{ def}}{=} \\\ \sum_{j} \delta_{j}^{l+1} \frac{\partial \sum_{k} w_{kj}^{l+1} a_{k}^{l} + b_{j}^{l+1}}{\partial z_{i}^{l}} \stackrel{a \text{ def}}{=} \sum_{j} \delta_{j}^{l+1} \frac{\partial \sum_{k} w_{kj}^{l+1} a(z_{k}^{l}) + b_{j}^{l+1}}{\partial z_{i}^{l}}. \\]

Because \\(a(z_{k}^{l})\\) only depends on \\(z_{i}^{l}\\) if \\(k = i\\), all but one of the partial derivatives become 0 when we differentiate. We obtain: \\[\delta_{i}^{l} = \sum_j \delta_{j}^{l+1} w_{ij}^{l+1} a'(z_{i}^{l}) \stackrel{\text{rearranging terms}}{=} \sum_j w_{ij}^{l+1} \delta_{j}^{l+1} a'(z_{i}^{l}). \quad \blacksquare \\] 

<a name="bp3"></a>**3. Rate of change of the cost w.r.t. any bias:** \\[\begin{equation} \tag{BP3} \frac{\partial C_x}{\partial b_{i}^{l}} = \delta_{i}^{l}. \end{equation} \label{eq:backprop3} \\]

**Matrix form:** \\[\frac{\partial C_x}{\partial b^l} = \delta^l.\\]

**Proof:** \\[\frac{\partial C_x}{\partial b_{i}^{l}} \stackrel{\text{chain rule}}{=} \sum_j \frac{\partial C_x}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial b_{i}^{l}}.\\] Since only \\(z_{i}^{l}\\) depends on \\(b_{i}^{l}\\): \\[\frac{\partial C_x}{\partial b_{i}^{l}} = \frac{\partial C_x}{\partial z_{i}^{l}} \cdot \frac{\partial z_{i}^{l}}{\partial b_{i}^{l}} \stackrel{z \text{ def}}{=} \frac{\partial C_x}{\partial z_{i}^{l}} \cdot \frac{\partial \sum_k w_{ki}^{l} a_{k}^{l-1} + b_{i}^{l}}{\partial b_{i}^{l}}.\\] Since none of the weights \\(w^{l}\\) and activations \\(a^{l-1}\\) depend on \\(b_{i}^{l}\\): \\[\frac{\partial C_x}{\partial b_{i}^{l}} = \frac{\partial C_x}{\partial z_{i}^{l}} \cdot 1 \stackrel{\delta \text{ def}}{=} \delta_{i}^{l}. \quad \blacksquare \\]

<a name="bp4"></a>**4. Rate of change of the cost w.r.t. any weight:** \\[\begin{equation} \tag{BP4} \frac{\partial C_x}{\partial w_{ij}^{l}} = a_{i}^{l-1} \delta_{j}^{l}. \end{equation} \label{eq:backprop4} \\]

**Matrix form:** \\[\frac{\partial C_x}{\partial w^{l}} = a^{l-1}(\delta^l)^T.\\]

**Proof:** \\[\frac{\partial C_x}{\partial w_{ij}^{l}} \stackrel{\text{chain rule}}{=} \sum_k \frac{\partial C_x}{\partial z_{k}^{l}} \cdot \frac{\partial z_{k}^{l}}{\partial w_{ij}^{l}}.\\] Since only the \\(j^{th}\\) input \\(z_{j}^{l}\\) depends on \\(w_{ij}^{l}\\), all the other terms \\((k \neq j)\\) become \\(0\\) when we differentiate: \\[\frac{\partial C_x}{\partial w_{ij}^{l}} = \frac{\partial C_x}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial w_{ij}^{l}} \stackrel{\delta \text{ def}}{=} \delta_{j}^{l} \frac{\partial z_{j}^{l}}{\partial w_{ij}^{l}} \stackrel{z \text{ def}}{=} \delta_{j}^{l} \frac{\partial \sum_k w_{kj}^{l} a_{k}^{l-1} + b_{j}^{l}}{\partial w_{ij}^{l}}.\\] Since only \\(w_{ij}^{l}\\) depends on \\(w_{ij}^{l}\\) (obviously), all terms where \\(k \neq i\\) become \\(0\\): \\[\frac{\partial C_x}{\partial w_{ij}^{l}} = \delta_{j}^{l} a_{i}^{l-1} \stackrel{\text{rearranging terms}}{=} a_{i}^{l-1} \delta_{j}^{l}. \quad \blacksquare \\]

This is it. We have derived the 4 equations of backpropagation - the first 2 equations allow us to efficiently calculate the errors in each layer, and the last 2 equations relate these errors to the derivatives of the cost function w.r.t. the parameters of the network. With these derivatives, we can construct the gradient \\(\nabla C_x\\) that is used in the gradient descent update rule \\(\eqref{eq:batch_gd}\\).<br><br>

## Furher reading
 
* 1: Rumelhart, D. Hinton, G. Williams, R. 1986. [Learning representations by back-propagating errors.](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)<br><br>

---
**Footnote:**

[^1]: 1: [Michael A. Nielsen, "Neural Networks and Deep Learning, Determination Press, 2015."](http://neuralnetworksanddeeplearning.com/)
[^2]: 2: This notation can easlily be extended to matrix form by omitting the subscripts. For example, \\(w^{l}\\) would be a matrix containing the weights connect the neurons in layer \\(l-1\\) to neurons in layer \\(l\\).
[^3]: 3: To be precise, the second assumption only makes sense in the context of neural networks. Thus, calling it a property of the activation function might not be entirely accurate, since the functions used in neural networks as activations are more general and are used in many different settings.     
[^4]: 4: The assumption that the input \\(z_{i}^{l}\\) influences the activation \\(a_{i}^{l}\\) **only** is true for [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [hyperbolic tangent](http://mathworld.wolfram.com/HyperbolicTangent.html) and [rectified linear units](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). However, for [softmax function](https://en.wikipedia.org/wiki/Softmax_function), this assumption does **not** hold. Therefore, the first 2 equations of backpropagation, as presented in this article, are not correct for the softmax function (or for any other function that invalidates the assumption for that matter).
[^5]: 5: [The four fundamental equations behind backpropagation.](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation)