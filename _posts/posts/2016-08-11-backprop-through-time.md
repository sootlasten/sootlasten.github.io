---
layout: post
title: "Backpropagation through time"
description: "Deriving backpropagation through time"
keyword: "recurrent neural network, RNN, neural network, backpropagation, deep learning, machine learning"
visible: 1
---

Backpropagation through time (BBTT) is simply the backpropagation algorithm applied in the context of [recurrent neural networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (RNN) to efficiently compute the gradient. In this post, we'll convince ourselves that BBTT is almost identical to regular backpropagation (which was derived in a [previous article](/2016/gd-backprop-derivation/#backpropagation)).

*(In the following analysis, the subscript \\(l(t)\\) means that we are dealing with a quantity in layer \\(l\\) and are considering timestep \\(t\\). Otherwise, the notation we are going to follow is formulated in the previous article.)*

In a regular feedforward neural network, each unit in layer \\(l\\) depends only on the units directly below it (i.e. on units in layer \\(l-1\\)). However, in a recurrent neural network, units in the hidden layers also depend on their previous activations: \\[a^{l(t)} = a(w^{l(t)}a^{l-1(t)} + w^{l(t-1)}a^{l(t-1)}),\\] where \\(a\\) represents a generic activation function and the bias is assumed the be implicit.

Because of this extra dependence in the hidden layers, we have to modify the equations of backpropagation derived earlier for a feedforward network. With that said, it might be surprising that the only equation we actually have to reconsider is [BP2](/2016/gd-backprop-derivation/#bp2). The other 3 equations ([BP1](/2016/gd-backprop-derivation/#bp1), [BP3](/2016/gd-backprop-derivation/#bp3), [BP4](/2016/gd-backprop-derivation/#bp4)) are still applicable in a RNN.

**Error \\(\bf \delta^{l(t)}\\) in terms of the error \\(\bf \delta^{l+1(t)}\\) in the next layer and the error \\(\bf \delta^{l(t+1)}\\) in the next timestep:** \\[\delta_i^{l(t)} = a'(z_i^{l(t)}) \left[ \sum_j w_{ij}^{l+1(t)} \delta_j^{l+1(t)} + \sum_k w_{ik}^{l(t+1)} \delta_k^{l(t+1)} \right].\\]

**Matrix form:** \\[\delta^{l(t)} = a'(z^{l(t)}) \odot \left[w^{l+1(t)} \delta^{l+1(t)} + w^{l(t+1)} \delta^{l(t+1)} \right].\\]

**Proof:**\\[\delta_i^{l(t)} \stackrel{\delta \text{ def}}{=}
\frac{\partial C}{\partial z_i^{l(t)}} \stackrel{\text{chain rule}}{=} 
\sum_j \frac{\partial C}{\partial z_j^{l+1(t)}} \cdot \frac{\partial z_j^{l+1(t)}}{z_i^{l(t)}} + \sum_k \frac{\partial C}{\partial 
z_k^{l(t+1)}} \cdot \frac{\partial z_k^{l(t+1)}}{\partial z_i^{l(t)}},\\] where the first sum is over the neurons in the \\((l+1)^{th}\\) layer at timestep \\(t\\), and the second sum is over the neurons in the \\(l^{th}\\) layer at timestep \\(l+1\\). To continue our derivation, we can use the definitons \\(\delta\\) and \\(z\\): 
\\[\delta_i^{l(t)} \stackrel{\delta \text{ def}}{=} \sum_j \delta_j^{l+1(t)} \frac{\partial z_j^{l+1(t)}}{\partial z_i^{l(t)}} + \sum_k \delta_k^{l(t+1)} \frac{\partial z_k^{l(t+1)}}{\partial z_i^{l(t)}} \stackrel{z \text{ def}}{=} \\\
\sum_j \delta_j^{l+1(t)} \frac{\partial \sum_m w_{mj}^{l+1(t)} a(z_m^{l(t)}) + b_j^{l+1(t)}}{\partial z_i^{l(t)}} + \sum_k \delta_k^{l(t+1)} \frac{\partial \sum_n w_{nk}^{l(t+1)} a(z_k^{l(t)}) + b_k^{l(t+1)}}{\partial z_i^{l(t)}}.\\] 

Because \\(a(z_m^{l(t)})\\) only depends on \\(z_i^{l(t)}\\) if \\(m = i\\) (an assumption we made about the activation function in the previous post) and similarly, \\(a(z_k^{l(t)})\\) only depends on \\(z_i^{l(t)}\\) if \\(n = i\\), all but 2 of the partial derivatives become \\(0\\) when we differentiate. We obtain: 
\\[\delta_i^{l(t)} = \delta_j^{l+1(t)} w_{ij}^{l+1(t)} a'(z_i^{l(t)}) + \sum_k \delta_k^{l(t+1)} w_{ik}^{l(t+1)} a'(z_i^{l(t)}) = \\\
a'(z_i^{l(t)}) \left[ \sum_j w_{ij}^{l+1(t)} \delta_j^{l+1(t)} + \sum_k w_{ik}^{l(t+1)} \delta_k^{l(t+1)} \right]. \quad \blacksquare
\\]



