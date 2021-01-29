---
layout: post
title: The Digital Image
date: 2020-11-17
description: "A digital image is an image composed of **picture elements**, also known as pixels, each representing a numeric quantity for its intensity or gray level value."
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
Image quantization refers to discretizing the intensity values of the image per channel. This is somewhat related to changing the *color depth* of each channel. If you recall, when we printed the values of the `img` a while ago, we found that we have an array with integer values. If you inspect it closely, you'll find that the integer values take on the values of 0 to 255 only. This is because our image currently has a quantization of `8-bits` per color channel which means that each channel can take on values of $2^8 = 256$ integers. This `8-bits` per channel correspond to the *24-bit True Color* depth, wherein we have a total of 16,777,216 colors to represent each pixel.

Let's look convert our titan image into an `8-bit` image, and see how this picture would look like in old-school consoles!

```python
# Initialize figure
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# Create bins for each channel
bins = np.linspace(0, img.max(), 8)

# Quantize the image
digitized = np.digitize(img, bins)
quantized = np.vectorize(bins.tolist().__getitem__)(digitized-1).astype(int)

# Show the images
axes[0].imshow(img)
axes[1].imshow(quantized)

# Set titles
axes[0].set_title("24-bit", fontsize=14, weight='bold')
axes[1].set_title("8-bit", fontsize=14, weight='bold');
```

![quantized-colossal-titan-img](/assets/images/quantized.png)

Well, it's not as `8-bit`-ety feel as like as those in the video games I used to play in our family computer! Haha. But this is to be expected, since our image is high resolution. Observe that the quantization does not change the resolution of the image but rather how well defined the colors are. In the `8-bit` image, we don't get that continous shadow on the face of the colossal titan as what we saw in the `24-bit` image.

Finally, let's look at how we grayscale an image.

### Converting to gray-scale

This is more straightforward than it looks, we just use the `rgb2gray` function of `skimage`.

```python
from skimage.color import rgb2gray
```

Then apply this function on our image.

```python
# Gray image
grayed = rgb2gray(img)

# Show image
io.imshow(grayed);
```

![grayed-image](/assets/images/grayed.png)

Now this is a bit of magic! Let's dig deep on how `skimage` ends up with this image. It turns out the transformation is very straightforward for converting colored images to grayscale, each pixel is just given a weighted sum depending on the intensities of each color channel. In particular, `skimage` uses the weights of **Cathode Ray Tube (CRT)** phospors as they mimic the human perception of red, blue, and green better than equal weights.

$$
Y = 0.2125 R + 0.7154 G + 0.0721 B
$$

## Conclusion

That's it, friends! We've covered some basic concepts of image processing on our very first blog post! Watch out for the next blog post! See you then!