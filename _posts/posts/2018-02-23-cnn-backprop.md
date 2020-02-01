---
layout: post
title: "Forward and backward propagations through a convolutional layer"
keyword: "gradient ascent"
visible: 1
---

In this post, we're going to derive the forward and backward passes through a convolutional layer and implement them in Python. For simplicity, we assume that the stride is of length 1.

Let's first lay down some notation (which unfortunately is a bit cluttered due to the fact that there are quite a few indices to keep track of):

* \\(z^{l}_{n,i,j,f}\\) - the input to unit \\(i,j,f\\) of layer \\(l\\) when training example number \\(n\\) is passed through the network. \\(i\\) indicates the index of the unit along the height of the feature map (y-dimension), while \\(j\\) indicates the index along the width (x-dimension), and \\(f\\) indicates the index along the depth (z-dimension);
* \\(x^{l}_{n,i,j,f} \\) - activation of feature \\(i,j,f\\) of layer \\(l\\) when training example number \\(n\\) is passed through the network;
* \\(w(f)_{i,j,c}^{l}\\) - the weight \\(i,j,c\\) of the \\(f\\)-th kernel in layer \\(l\\);
* \\(b(f)^{l}\\) - the bias of the \\(f\\)-th kernel in layer \\(l\\).<br><br>


## Forward propagation

#### Math 

Suppose layer \\(l\\) with \\(D\\) channels is convolved with the \\(f\\)-th kernel \\(w(f)^{l+1}\\) of height \\(H_k\\), width \\(W_k\\) and depth \\(D\\). Then the pre-activation weighted sum \\(z^{l+1}_{n,i,j,f}\\) of the next layer can be computed as follows:

$$ \begin{equation} z^{l+1}_{n, i, j, f} =  \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^D x_{n, i+m-1, j+n-1, d}^{l} w(f)_{m,n,d}^{l+1} + b(f)^{l+1} \end{equation} \label{eq:forwardconv}.$$

Intuitively, we can arrive at the previous equation by picking an arbitrary unit in layer \\(l+1\\) and noting which inputs in the previous layer \\(l\\) were responsible for generating it. Following that line of though, we arrive at the conclusion that unit \\((i,j)\\) in layer \\(l+1\\) is affected by a rectangular region of size HxW in the previous layer whose top-left corner is also at position \\((i,j)\\).<br><br>

#### Code

In a fully connected neural network, doing a pass through a dense layer amounts to carrying out just a single matrix multiplication. This is good news, because that means we can make use of the [fast matrix multiplication operations](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/) scientific programmers have been optimizing since the advent of computing. 

Luckily for us, it turns out that convolving a bunch of kernels across an image can be done in a single matrix multiplication as well. The setup is rather straightforward: one needs to construct a matrix whose columns are composed of patches of images that would be multiplied with a kernel. An operation that does this procedure is called **im2col**. Now, if you multiply this matrix on the left with another matrix whose rows carry all the kernels of a specific layer, you'll have just done a complete convolution. A more thorough discussion on im2col can be found [here](http://cs231n.github.io/convolutional-networks/#im2col).

Here's a <u>heavily</u> commented code snippet in Python carrying out im2col:

```python
def im2col(imgs, ker_shape, stride):
    """
    Parameters
    ----------  
    imgs: np.ndarray
        4D tensor of shape (B, H, W, C), where B: batch_size, H: feature map height, 
        W: feature map width, C: number of channels of feature map
    ker_shape: tuple
        2D tuple, where the first element is kernel height (kh), 
        while the second is kernel width (kw)
    stride: int
        Convolution stride
    
    Returns
    -------
    np.ndarray
        A 3D matrix of size (N, kh*kw*C, -1) 
    """
    _, xh, xw, xc = imgs.shape  # input featre map dimensions
    kh, kw = ker_shape          # kernel dimensions

    # Indices of the first position in the feature map kernels are applied.
    # For example, if the input feature map has W=30,C=3, and the kernel 
    # is a square of size 3, then:
    # >>> print(first_ker_pos)
    # >>> array([[  0,   3,   6],
    #            [ 90,  93,  96],
    #            [180, 183, 186]])
    first_ker_pos = (np.arange(kh)*xw*xc)[:, np.newaxis] + np.arange(kw)*xc
    
    # Add channel information to the previous matrix, such that the new 
    # 3D matrix is of size(kh, kw, C).
    # >>> print(reshaped_first_ker_pos)
    # >>> array([[[  0,   1,   2],
    #             [  3,   4,   5],
    #             [  6,   7,   8]],

    #            [[ 90,  91,  92],
    #             [ 93,  94,  95],
    #             [ 96,  97,  98]],

    #            [[180, 181, 182],
    #             [183, 184, 185],
    #             [186, 187, 188]]])
    reshaped_first_ker_pos = first_ker_pos[:, :, np.newaxis] + np.arange(xc)

    # Indices of the upper left corners of the positions that are dot producted with a kernel
    # For example, if H,W = 30, kernel is a square of size 3, and stride=1, then:
    # >>> print(hpos)
    # >>> [0, 1, 2, ..., 27]
    # >>> print(wpos)
    # >>> [0, 1, 2, ..., 27]
    hpos, wpos = np.arange(0, xh-kh+1, stride), np.arange(0, xw-kw+1, stride)

    # Construct a 2D matrix where the first element (h,w) is the index of the first 
    # channel value of unit (h,w) if the original 3D input feature map were flattened
    # to a 1D array of size of size HxWxC.
    # Following our example:
    # >>> print(pos[0])
    # >>> array([0, 3, 6, 9, 12, ..., 81])
    # >>> print(pos[1])
    # >>> array([90, 93, 96, 99, 102, ..., 171])
    pos = hpos[:, np.newaxis]*xw*xc + wpos*xc

    # Construct a matrix of indices, where each column is of length kh*kw*C, with the 
    # i-th column carrying the indices of the units of the orig. feature map that would 
    # fall under the kernel in the i-th "timestep" (if kernel is thought to be applied 
    # iteratively from top left corner in clockwise fashion). Note that as before, 
    # the indices are all w.r.t. a flattened feature map.
    # Following our example:
    # >>> print(im2col_indices.shape)
    # >>> (27, 784)
    # >>> print(im2col_indices[:, 0])
    # >>>  array([  0,   1,   2,   3,   4,   5,   6,   7,   8,  90,  91,  92,  93,
    #               94,  95,  96,  97,  98, 180, 181, 182, 183, 184, 185, 186, 187, 188])
    im2col_indices = reshaped_first_ker_pos.ravel()[:, np.newaxis] + pos.ravel()
    
    # Flatten the image and just "fill" the matrix of indices from the previous line
    # with the real feature values from the feature map. Note that when we flatten the 
    # feature map, we retain a 2D matrix of size (N, xh*xw*xc), since we are preserving
    # the batch dimension.
    flat_img = imgs.reshape(-1, xh*xw*xc)
    return np.take(flat_img, im2col_indices, axis=1)
```

<br>
Here's how one would go about utilizing the above procedure to do a full forward pass in a convolutional layer:

```python
def forward(x, w, ker_shape, stride, out_shape):
    """
    Parameters
    ----------
    x: np.ndarray
        4D tensor of shape (B, H, W, C) representing the input to the convolutional layer.
        Note that we assume the input is padded already. 
    w: np.ndarray
        2D tensor of shape (kn, kh*kw*C), where the i-th row contains the weights of the 
        i-th kernel, and kn is the number of kernels.
    ker_shape: tuple
        2D tuple, where the first element is kernel height (kh), while the second is 
        kernel width (kw)
    stride: int
        Convolution stride
    out_shape: tuple
        3D tuple of form (H_out, W_out, C) representing the shape of the output 
        feature map after the convolutional layer.
    
    Returns
    -------
    np.ndarray
        A 4D matrix of the output feature maps. The matrix is of size (B, H_out, W_out, kn) 
    """
    batch_size, xh, xw, _ = x.shape
    kh, kw = ker_shape

    # Apply the im2col procedure on the input feature maps.
    xcol = im2col(x, ker_shape, stride)

    # Convolve the kernels over the input feature maps. Notice
    # that we can do this relatively complex procedure with just
    # a single matrix multiplication. We need to use numpy's 
    # tensordot because the 0-th dimension of xcol is the batch dim.
    # The resulting matrix has the shape (kn, batch_size, H_out*W_out),
    # where kn is the number of kernels.
    next_fmap =  np.tensordot(w, xcol, (1, 1))
    
    # Expanding the last dimension from flattened form back to an image.
    # The shape is now (kn, batch_size, H_out, W_out).
    next_fmap = next_fmap.reshape(fmap.shape[:-1] + out_shape[:-1])

    # Swapping axes such that we'd arrive at the standard format of 
    # (batch_size, H_out, W_out, kn).
    return next_fmap.transpose(1, 2, 3, 0)
```
<br>

## Backward propagation

#### Math

The derivative of the loss \\(L\\) w.r.t. to weight \\(w(f)_{i,j,c}^{l+1}\\) can be computed as follows (\\(H^{l+1}_o\\) and \\(H^{l+1}_w\\) are the width and height of the activation map of layer \\(l+1\\), and \\(N\\) is the number of training exmaples): 

$$ \begin{equation} \frac{\partial L}{\partial w(f)_{i,j,c}^{l+1}} = \sum_{n=1}^N \sum_{a=1}^{H^{l+1}_o} \sum_{b=1}^{W^{l+1}_o} \frac{\partial L}{\partial z_{n,a,b,f}^{l+1}} \frac{\partial z_{n,a,b,f}^{l+1}}{\partial w(f)^{l+1}_{i,j,c}}. \end{equation} \label{eq:backconv} $$

When you inspect the above equation, you see that we sum across the pre-activations of the next layer, instead of taking the average. This [might seem counter-intuitive](https://github.com/BVLC/caffe/issues/3242) at first, but mathematically, it is just what the chain rule tells us to do. For example, consider a bivariate nested function: \\(L(z_1(w), z_2(w))\\). According to the chain rule, the derivative w.r.t. the parameter \\(w\\) is then \\(\frac{L(z_1(w), z_2(w))}{\partial w} = \sum_{i=1}^2 \frac{\partial L(z_i(w))}{\partial z_i(w)} \frac{\partial z_i(w)}{\partial w}\\). The connection to convnets is obvious, and made explicit by the notation.

What is left is to derive the 2 derivatives in the above equation. Let's start with the leftmost one, the **derivative of the loss w.r.t. input \\(x\\) to the conv layer** in terms of output \\(z\\) of the same conv layer (\\(H_k,W_k\\) are the height and width of the kernel, \\(D_o^{l+1}\\) is the depth of layer \\(l+1\\)), for which we first derive an intermediate quantity:

$$ \begin{equation} \frac{\partial L}{\partial x_{n,i,j,f}^{l}} = \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^{D_o^{l+1}} \frac{\partial L}{\partial z^{l+1}_{n, i-m+1, j-n+1, d}} \frac{\partial z^{l+1}_{n, i-m+1, j-n+1, d}}{\partial x^l_{n,i,j,f}}. \end{equation} \label{eq:dL_dx} $$

It is interesting to compare this to the forward propagation equation \eqref{eq:forwardconv}, where we noted that unit \\((i,j)\\) in layer \\(l+1\\) is affected by a rectangular region of size \\(H_k \times W_k\\) in the previous layer whose **top-left corner** is also at position \\((i,j)\\). In the backward case, we see the exact opposite: unit \\((i,j)\\) in layer \\(l\\) is affected (or "is affecting", depending on the order of causality) a rectangular region of size \\(H_k \times W_k\\) in the next layer whose **bottom right corner** is also at position \\((i,j)\\).

Continuing with the derivation, we can substitute equation \eqref{eq:forwardconv} into into equation \eqref{eq:dL_dx}:

$$ \frac{\partial L}{\partial x_{n,i,j,f}^{l}} = \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^{D_o^{l+1}} \frac{\partial L}{\partial z^{l+1}_{n, i-m+1, j-n+1, d}} \\ \cdot \frac{\partial}{\partial x^l_{n,i,j,f}} \Big[\sum_{a=1}^{H_k} \sum_{b=1}^{W_k} \sum_{c=1}^D x_{n, (i-m+1)+a-1, (j-n+1)+b-1, c}^{l} w(d)_{a,b,c}^{l+1} + b(d)^{l+1} \Big] \\ 
  = \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^{D_o^{l+1}} \frac{\partial L}{\partial z^{l+1}_{n, i-m+1, j-n+1, d}} \frac{\partial x_{n, i, j, c}^{l}}{\partial x_{n,i,j,f}^l} w(d)_{m,n,f}^{l+1} \\ = \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^{D_o^{l+1}} \frac{\partial L}{\partial z^{l+1}_{n, i-m+1, j-n+1, d}}w(d)_{m,n,f}^{l+1}.$$
  
Now, calculating the error in one layer in terms of the error in the next layer is straightforward:

$$ \begin{equation} \frac{\partial L}{\partial z_{n,i,j,f}^l} = \frac{\partial L}{\partial x_{n,i,j,f}^l} \frac{\partial x_{n,i,j,f}^l}{\partial z_{n,i,j,f}^l} \\ =
\frac{\partial x_{n,i,j,f}^l}{\partial z_{n,i,j,f}^l} \sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^{D_o^{l+1}} \frac{\partial L}{\partial z^{l+1}_{n, i-m+1, j-n+1, d}} w(d)_{m,n,f}^{l+1}, \end{equation} \label{eq:backerr} $$

where the first derivative is of the activation function w.r.t. its input and is easy to compute.<br><br>

Now, let's focus on the remaining term in equation \eqref{eq:backconv}, the **derivative of conv layer output w.r.t. kernel weight**:

$$\frac{\partial z_{n,a,b,f}^{l+1}}{\partial w(f)^{l+1}_{i,j,c}} = \frac{\partial}{\partial w(f)^{l+1}_{i,j,c}} \Big[\sum_{m=1}^{H_k} \sum_{n=1}^{W_k} \sum_{d=1}^D x_{n,a+m-1, b+n-1, d}^{l} w(f)_{m,n,d}^{l+1} + b(f)^{l+1} \Big] \\ 
= \frac{\partial}{\partial w(f)^{l+1}_{i,j,c}} \Big( x_{n, a+i-1, b+j-1, c}^{l} w(f)_{i,j,c}^{l+1} \Big) = x_{n, a+i-1, b+j-1, c}^{l}.$$

Substituting it back into the original derivative, we obtain the final solution:

$$ \begin{equation} \frac{\partial L}{\partial w(f)_{i,j,c}^{l+1}} = \sum_{n=1}^N \sum_{a=1}^{H^{l+1}_o} \sum_{b=1}^{W^{l+1}_o} \frac{\partial L}{\partial z_{n,a,b,f}^{l+1}} x_{n,a+i-1, b+j-1, c}^{l}, \end{equation} $$

where the derivative of the loss w.r.t. input to layer \\(l+1\\) was calculated above.<br><br>

#### Code

Note that in the following code snippet, we are not actually calculating \\(\frac{\partial L}{\partial z} = \frac{\partial L}{\partial x^l} \frac{\partial x^l}{\partial z^l}\\) (as per equation \eqref{eq:backerr}), where \\(z\\) is the input to layer \\(l\\) and \\(x^l\\) is the same input passed through the activation function. This is because in modular backprop, the convolutional layer would only take care of the convolution part, and we'd have another layer for actually applying the activation function, yielding the conv layer responsible only for the derivative \\(\frac{\partial L}{\partial x^l}\\).

```python
def backward(dz, x, w, ker_shape, stride, out_shape, padding='same'):
    """ 
    Parameters
    ----------
    dz: np.ndarray
        4D tensor of shape (B, H_out, W_out, D_out) holding the gradients of the loss w.r.t.
        the output of this conv layer. This is what is given you when doing modular backprop.
    x: np.ndarray
        4D tensor of shape (B, H, W, C) representing the input to the convolutional layer.
        Note that we assume the input is unpadded.
    w: np.ndarray
        2D tensor of shape (N, kh*kw*C), where the i-th row contains the weights of the 
        i-th kernel, and N is the number of kernels.
    ker_shape: tuple
        2D tuple, where the first element is kernel height (kh), while the second is 
        kernel width (kw)
    stride: int
        Convolution stride
    out_shape: tuple
        3D tuple of form (H_out, W_out, C) representing the shape of the output 
        feature map after the convolutional layer.
    padding: string
        Specifies the padding type. For example, 'same' is commonly used, which means
        that the spatial size of the output of the conv layer should be the same as the
        input's. 
    
    Returns
    -------
    wgrad: np.ndarray
        A 2D matrix of gradients of the loss w.r.t. kernel weights. The matrix is of
        size (oc, kh*kw*C), where C is the depth of the input feature map.
    dx: np.ndarray
        A 4D matrix of gradients of the loss w.r.t. the input feature map of this layer. 
        The matrix is of size (B, H, W, C).
    """
    # Pad the input.
    x_padded = pad(x, out_shape, ker_shape, stride, padding)
    
    batch_size, xh, xw, xc = x_padded.shape
    oh, ow, oc = out_shape
    kh, kw = ker_shape

    # WEIGHT UPDATES (equation 5 in the text)
    # ---------------------------------------

    # Flattening the 2D "image" in the middle, such that the new 
    # shape of the tensor holding gradients is (batch_size, oh*ow, oc)
    dz_flat = dz.reshape(batch_size, -1, oc)
    
    # Im2col on the input feature map. Note that in a real implementation,
    # this matrix would have been stored in the forward pass, but is re-done
    # here explicitly for illustrative purposes.
    xcol = im2col(x, kernel_shape, stride)

    # Doing a dot product between the last dimension of 'xcol' and the middle 
    # dimension of 'dz_flat'. This amounts to taking the pairwise products between
    # each activation map in the input feature map and the gradient tensor, and 
    # summing the results, as per equation 5 in the text, but without summing across 
    # tha batches.  The resulting tensor is of size (batch_size, kh*kw*xc, oc).
    grads = np.einsum('bwp,bpc->bwc', xcol, dz_flat)

    # Now we also sum across the batches and swap the axes such that the gradient
    # matrix would be of size (oc, kh*kw*xc).
    wgrad = np.sum(grads, axis=0).swapaxes(0, 1)
        

    # ERRORS (half of equation 4 in the text,
    # because as was said in the text, the layer 
    # is only responsible for calculating dL/dx).
    # -------------------------------------------
    
    # Pad the gradient tensor
    dz_padded = pad(dz, out_shape, ker_shape, stride, padding)

    # Because the gradient tensor is like a feature map itself, we can apply
    # im2col to it. The resulting tensor is of size (batch_size, kh*kw*oc, oh*ow).
    dzcol = im2col(dz_padded, ker_shape, stride)

    # Reshape the matrix holding the kernel weights such that it would be of 
    # size (xc, kh*kw*oc).
    wback = w.reshape(oc, kh, kw, xc).swapaxes(0, -1).reshape(xc, kh*kw*oc)

    # Dot product between the kernel matrix and the incoming gradients, as per
    # equation 4. The resulting tnesor is of size (xcol, batch_size, H*W), where
    # H and W are the height and width of the unpadded feature map in this layer.
    dx =  np.tensordot(wback, dzcol, (1, 1))

    # Expand the last dimension of the previous matrix, so it would be a 2D
    # feature map. The new size is then (xcol, batch_size, H, W).
    dx = dx.reshape(dx.shape[:-1] + tuple(self.prev.out_shape[:-1]))
    
    # Swap axes, the transform the tensor holding the gradients into the correct shape:
    # (batch_size, H, W, xc).
    dx = dx.transpose(1, 2, 3, 0)

    return wgrad, dx 
```
