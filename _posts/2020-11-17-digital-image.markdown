---
layout: post
title: The Digital Image
date: 2020-11-17
description: 
image: /assets/images/colossal-bw.png
author: Leo Lorenzo II
tags: 
  - skimage
  - Introduction to Image Processing
---

"A digital image is an image composed of **picture elements**, also known as pixels, each representing a numeric quantity for its intensity or gray level value."

This simple sentence might seem trivial to some, but as my classmate in our IIP class pointed out, this opens up a whole new world of perspective.

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post, we look at how we can read, manipulate, and store an image using `python`.

## Python as an image processing tool?

When we think of image processing tools, we think about **Photoshop**, **LightVSCO**, and **Instagram** filters that you love to tinker on. Little do we know that behind the filters and the buttons and screws of these filters is an intricate Mathematics and well-defined matrix multiplication operations.

Here, we will demonstrate the mathematics behind some of those operations using `python`. Yes, `python` the ultimate programming language that can do almost anything (almost!). We'll do a simple operation of opening an image, then performing some operations such as `sampling`, `quantization`, and converting images to a `grayscale`. These operations are trivial but we're going to dig deep on what happens when we do these operations and demonstrate them using `python`.

Let's tackle the first question in everybody's mind. How do you open an image using `python`? Well, in photoshop, paint, or whatever, what we do is open an image using the graphic interface of that application, right? For `python`, we do the same, but we codify it, of course!

### Reading an image using `Python`

Let's start with a band! Let's use the image of the colossal titan shown in the preview of this blog post.

*Image taken from: https://static2.cbrimages.com/wordpress/wp-content/uploads/2019/10/Attack-on-Titan-Colossal-Titan-Cropped.jpg*

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
img = io.imread('lena.png')
```

The above cell reads the `png` image stored in our directory as a variable `img`. But what does this `img` look like? Let's print this in our notebook to see.

```python
img
```

    array([[[ 57,  48,  31],
            [ 26,  14,   0],
            [219, 201, 165],
            ...,
            [176, 169, 150],
            [132, 131, 113],
            [ 41,  42,  24]],
          [[ 57,  48,  31],
            [ 26,  14,   0],
            [219, 201, 165],
            ...,

As you can see, the image of colossal titan is represented as numbers. Wherein each number, by default, corresponds to the intensity values of the **3** color channels, **RGB (Red-Green-Blue)**. There are other ways to represent this three channels which include the **HSV (Hue-Saturation-Value)** and the **CMYK (Cyan-Magenta-Yellow and Key)**.

Let's show the image on our `python` notebook using `skimage`.

```python
io.imshow(img);
```
![png](/assets/images/io.imshow.png)

There it is! The scariest titan we've ever seen!

Now, let's perform several operations on this image of the titan.

### Sampling an image

Sampling refers to taking the value of the image at regular spatial intervals. The length of the intervals define the resulting spatial resolution of the image. 

In `skimage`, this operation can be done using the function `downscale_local_mean`

```python
from skimage.transform import downscale_local_mean
```

Since, the colossal titan is too big and scary, let's try to downsample it and look at the effect of this operation on the image.

```python
# Set downsampling factors
factors = [32, 8, 1]

# Initialize figures
fig, axes = plt.subplots(1, 3, figsize=(16, 9))

# Iterate through axis and factors
for ax, factor in zip(axes, factors):
    # Get downsampled image
    cur_img = downscale_local_mean(img, 
                                   factors=(factor, factor, 1)).astype(int)
    ax.imshow(cur_img)
    ax.set_title(f"Factor = {factor}")
```

![downsampled-images](/assets/images/downsampling.png)

As you can see, as we increase the downsampling factor, the image becomes more pixelated. Additionally, if we notice the dimensions in the `x-axis`, the total number of pixel dimensions become smaller. Which means that when we use the function `downscale_local_mean`, what happens is that it groups pixels that are in neighbor with each other, then takes the mean pixel value of that location, effectively reducing the dimension of the image.

This operation might be similar when you decrease the size of an image in powerpoint by using the resizing tool.

Next, let's look at the quantization operation.

### Quantization of an image



### Separating the color channels