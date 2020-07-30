---
layout: post
title: "A day spent debugging a neural network"
visible: 0
---

I have a deep-learning-based segmentation model that takes an RGB panorama image as an input and outputs a segmentation map that's eight times smaller in both the width and height dimensions. This coarse output gives a reasonably good general overview of the scene, but is insufficient when one wants to, for example, precisely delineate the edges of a sidewalk. Thus motivated, I set out to increase the output map by a factor of two. I thought it would take me five minutes. It took the whole day.

At first, everything went like clockwork. I attached an additional deconvolution layer to the end of the model, changed a configuration parameter pertaining to input downscaling from eight to four, and the next thing you know, the network was training and the loss was decreasing. Calling it a job well done, I left the neural network to do its thing and moved on to other projects, promising to check back an hour later for the results.

But when I did, I was confronted with this:

```bash
/opt/conda/conda-bld/pytorch_1523244252089/work/torch/lib/THCUNN/ClassNLLCriterion.cu:101: 
void cunn_ClassNLLCriterion_updateOutput_kernel(Dtype *, Dtype *, Dtype *, long *, Dtype *, int, int, int, int, long) 
[with Dtype = float, Acctype = float]: block: [0,0,0], thread: [23,0,0] Assertion `t >= 0 && t < n_classes` failed.

Traceback (most recent call last):
  ... [removed for brevity]
  File "/home/sten/segmentation-dnn/model.py", line 102, in numpy_to_var
    v = v.cuda()
  File "/home/sten/miniconda2/envs/pytorch_env/lib/python3.6/site-packages/torch/autograd/variable.py", line 298, in cuda
    return CudaTransfer.apply(self, device, async)
  File "/home/sten/miniconda2/envs/pytorch_env/lib/python3.6/site-packages/torch/autograd/_functions/tensor.py", line 201, in forward
    return i.cuda(async=async)
  File "/home/sten/miniconda2/envs/pytorch_env/lib/python3.6/site-packages/torch/_utils.py", line 69, in _cuda
    return new_type(self.size()).copy_(self, async)
RuntimeError: cuda runtime error (59) : device-side assert triggered at /opt/conda/conda-bld/pytorch_1523244252089/work/torch/lib/THC/generic/THCTensorCopy.c:21
```
<br>
From CUDA-s failed assertion ``t >= 0 && t < n_classes``, I saw that the problem is that I'm testing the network against a label whose value is outside the range of the number of softmax units I have in the final layer. However, Python's traceback seems to indicate that the problem arose from trying to move the variable `v` onto GPU in the function `numpy_to_var`. What's going on? Why does CUDA tell us one thing, Python another?

The conundrum lies in the fact that PyTorch [executes CUDA calls asynchronously](seems to indicate that the problematic call was in the function `numpy_to_var`.) - when a PyTorch program calls a CUDA operation, it doesn't sit still, waiting until the computation on the GPU finishes, but continues executing the Python program until it actually needs a result. While this gives great performance, it can result in confusing tracebacks, because by the time the CUDA operation fails, the Python program has long gone on to do other stuff.

To get an instructive traceback, one must set the environment variable `CUDA_LAUNCH_BLOCKING=1`, which disables PyTorch's asynchronous execution mode. Doing that, the problematic call in Python reveals itself:

```bash
/opt/conda/conda-bld/pytorch_1523244252089/work/torch/lib/THCUNN/ClassNLLCriterion.cu:101: 
void cunn_ClassNLLCriterion_updateOutput_kernel(Dtype *, Dtype *, Dtype *, long *, Dtype *, int, int, int, int, long) 
[with Dtype = float, Acctype = float]: block: [0,0,0], thread: [23,0,0] Assertion `t >= 0 && t < n_classes` failed.

Traceback (most recent call last):
  ...
  File "/home/sten/segmentation-dnn/loss.py", line 295, in get_loss
    loss = F.cross_entropy(pred, gt)
  File "/home/sten/miniconda2/envs/pytorch_env2/lib/python3.6/site-packages/torch/nn/functional.py", line 1161, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)
  File "/home/sten/miniconda2/envs/pytorch_env2/lib/python3.6/site-packages/torch/nn/functional.py", line 1052, in nll_loss
    return torch._C._nn.nll_loss(input, target, weight, size_average, ignore_index, reduce)
RuntimeError: cuda runtime error (59) : device-side assert triggered at /opt/conda/conda-bld/pytorch_1523244252089/work/torch/lib/THCUNN/generic/ClassNLLCriterion.cu:113
```
<br>
This is truly confounding, though. The model has been trained with the higher resolution output for hundreds of times with no issues. How can merely changing the size of ground truth labelmaps take them out of the correct range?

After banging my head against the table for ten hours, I hit upon an answer. First, I have several datasets, gathered at different times, that I merge into one for training the network. One of them, call it dataset A, has 1 class, `unlabelled_region`, more than the others, and the indices of the labels it has in common with the other datasets are shifted to the right by one. I.e., the road class for dataset A is associated with the index 1 of the label map, while in the other datasets, its index is 0, etc. To make dataset A compatible with the other datasets, I merged the `unlabelled_region` with the `miscellaneous` class, and shifted all the indices left by one. 

Now, since the `miscellaneous` class is associated with the last index, the procedure can be accomplished with a 2-liner:
```python
label_img -= 1
label_img[label_img == -1] = num_classes - 1
```
<br>
And herein lies the bug. What I failed to recognize was that `label_img` is a 2d matrix of 8-bit unsigned integers, of type `np.uint8`. Trying to set a negative value for this matrix results in the value being circled up to 255, the highest number for this data type. Thus, instead of setting the `unlabelled_region` class labels to the index `num_classes - 1`, I set them to 255. Out of range, indeed! I should have done this instead:

```python
label_img -= 1
label_img[label_img == 255] = num_classes - 1
```
<br>
But wait a minute, why didn't the error come up while training the model with higher resolution output? The answer is as satisfying as it is funny: thanks to cold, hard luck.

When resizing the grayscale label image, where each position holds a class index from ``{0, 1, ..., `num_classes - 1`}``, it's first transformed into a stack of `num_classes` binary sheets, where the `i`-th sheet is white in regions which corresponded to the `i`-th segmentation class, and black otherwise. Subsequently, each of these sheets is downscaled separately using opencv-s `INTER_AREA` resizing method to match the desired output size of the neural network. To get the final downscaled labelmap, an `argmax` is taken across the sheets. Here's the process in code:

```python
def mask_resize(img, res_wh):
    res_wh = (*res_wh,)
    max_label = np.max(img)
    planes = [cv2.resize((255 * (img == v)).astype(np.uint8), res_wh, interpolation=cv2.INTER_AREA) for v in range(max_label+1)]
    stacked_planes = np.stack(planes)
    argmax = np.argmax(stacked_planes, axis=0)
    argmax = argmax.astype(np.uint8)
    return argmax
```
<br>
It just so happened that when the output resolution was lower (i.e. `res_wh` was smaller), the white regions in the sheets were such that upon resizing and argmaxing, the results came out correct. However, when the resolution is increased, opencv interpolates over more regions, and the incorrect values don't escape the argmax operator. What a bug...

