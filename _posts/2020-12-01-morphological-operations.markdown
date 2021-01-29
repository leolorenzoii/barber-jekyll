---
layout: post
title: Morphological Operations
date: 2020-12-01
description: When your Mom tells you to clean your room or your home, what do you say to her? How do you do it? What does this have to do with image processing?
image: /assets/images/3-morphological-operations/cleaned.png
author: Leo Lorenzo II
tags: 
  - skimage
  - morphological operations
---

When your Mom tells you to clean your room or your home, what do you say to her? How do you do it? What does this have to do with image processing?

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post, we'll explore the world of morphological operations and their use in image processing, i.e., CLEANING!

## What are Morphological Operations?

So that is what morphological operations is all about, cleaning? Well at least you can think of it that way. Haha. Technically speaking, morphological operations alter the shape of or morphology of the features in an image such as boundaries, skeletons, etc. In doing so, we use what we call a **structuring element**, which is essentially the *"broom"* that we will use in cleaning or altering our image.

Morphological operations were originally defined in the context of a *binary image* - a case where you have background pixels of `0` value, and a foreground (or object of interest) with a pixel value of `1`. The concept of morphological operation can easily be extended for grayscale images but for now, let us consider the binary case for easy demonstration.

There are two basic morphological operations in which other more complex procedures based on:

1. Erosion
2. Dilation

**Erosion** essentially uses the structuring element to reduce the shape contained in the image. Mathematically speaking, erosion of a given image $A$ with a structuring element $B$ are all the pixels $x$ such that $B$ is still contained in the image $A$. Thus, the effect of erosion is to reduce the total foreground pixels in our image.

In contrast, **Dilation** tends to enlarge the area of the foreground pixels. Dilation does this by searching for pixel locations where the foreground pixel and the structuring element may overlap by one pixel. Mathematically, this is defined as the set of all pixels such that the structuring element $B$ and foreground pixels $A$ overlap by at least one non-zero element.

Let's look at these two morphological operations in action:

```python
# Importing some libraries that we will use
from skimage.morphology import diamond, ball, dilation, erosion
import numpy as np
import matplotlib.pyplot as plt
```

### Erosion

Let's take the case where our original image is of diamond shape:

```python
# Define foreground image
img = np.pad(diamond(6), 2)
plt.imshow(img, cmap='plasma');
```

![diamond](/assets/images/3-morphological-operations/diamond.png)

Afterwards, let's set our structuring element `selem` as a ball shape:

```python
# Define structuring element
selem = np.pad(ball(4)[2], 4)
plt.imshow(selem, cmap='plasma');
```

![ball](/assets/images/3-morphological-operations/ball.png)

Let's perform the erosion operation, and inspect the result:

```python
# Perform erosion
eroded = erosion(img, selem)

# Plot result
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
axes[0].imshow(img, cmap='plasma')
axes[1].imshow(selem, cmap='plasma')
axes[2].imshow(eroded + img, cmap='plasma')

# Set labels
axes[0].set_title("Image", fontsize=14, weight='bold')
axes[1].set_title("Structuring Element", fontsize=14, weight='bold')
axes[2].set_title("Erosion", fontsize=14, weight='bold');
```

![erosion](/assets/images/3-morphological-operations/erosion.png)

Here, the resulting image after erosion is the yellow region in the rightmost figure. Notice that the total area of the foreground image decreases as the result of erosion. This is because, what erosion does is really remove all the pixels in the foreground in which our structuring element does not fit.

Let's look at this more in action, check out the animation below:

![erosion-animation](/assets/images/3-morphological-operations/erosion.gif)

Notice that the pixels being retained by the operation are those pixels wherein the structuring element fits completely with the foreground image. As a result, some of the original pixel of the foreground image is eroded away due to this operation.

### Dilation

Next, let's look at dilation. 

Again, let's use a diamond as our foreground image:

```python
# Define foreground image
img = np.pad(diamond(6), 3)
plt.imshow(img, cmap='plasma');
```

![diamond-2](/assets/images/3-morphological-operations/diamond-2.png)

While for this case, let's use a cross-shaped structuring element for simplicity.

```python
# Define structuring element
selem = np.pad(ball(1)[1], 8)
plt.imshow(selem, cmap='plasma');
```

![cross](/assets/images/3-morphological-operations/cross.png)

Let's perform the dilation operation and inspect the result:

```python
# Perform erosion
dilated = dilation(img, selem)

# Plot result
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
axes[0].imshow(img, cmap='plasma')
axes[1].imshow(selem, cmap='plasma')
axes[2].imshow(dilated + img, cmap='plasma')

# Set labels
axes[0].set_title("Image", fontsize=14, weight='bold')
axes[1].set_title("Structuring Element", fontsize=14, weight='bold')
axes[2].set_title("Dilation", fontsize=14, weight='bold');
```

![dilation](/assets/images/3-morphological-operations/dilation.png)

Here, the resulting image is the pink image on the rightmost plot. Notice that as a result of the dilation operation, the total area of the foreground pixels increased. This is because in contrast to erosion, the effect of dilation is to add pixel to any location at which the structuring element and the original image touches by at least 1 pixel. This effectively increases the total area of the resulting image. Let's look at this operation in action as shown by the animation below:

![dilation animation](/assets/images/3-morphological-operations/dilation.gif)

Notice that everytime the structuring element and the original image overlaps, a pixel is added on the image.

### Other Morphological Operations

Other useful morphological operations that is usually used for preprocessing images are as follows:

1. Closing
2. Opening
3. White tophat
4. Black tophat
5. Area closing
6. Area opening

**Closing** is defined as a dilation followed by an erosion using the same structuring element. Closing is usually done to remove small dark spots and essentially "connect" and "close" foreground pixels.

**Opening** is defined as an erosion followed by a dilation using the same structuring element. Opening is usually done to remove small white spots and essentially "open" and "disconnect" foreground pixels.

**Black tophat** is closing minus the original image. This means that the dark spots smaller than the structuring element will be returned. This is usefull when isolating small dark elements in the image.

**White tophat** is defined as the image minus its morphological opening. This means that the small white spots will be returned by this operation. This is useful when isolating small white elements in the image.

Finally, **area closing** and **area opening** are two variants of opening and closing, but instead of giving a structuring element, an area is instead specified in the parameter. Thus, small dark or bright spots will be removed depending if area closing or area opening operation is used. Note that the objects in the image are designated as the connected components in the image.

Throughout our blogs, you might see us using these operations once in a while. Note that each has their own strength and weaknesses and may be appropriate to use in some special cases. The cool thing (but also difficult thing) about image processing is that there is no clear cut on what procedure to use per case. But as long as we understand the fundamentals of morphological erosion (erosion and dilation), we can have an idea of what the effects of these operations will be on our image.

### Morphological Operation Demo

Let's demonstrate our morphological operations to a generic cleaning problem of optical character recognition:

![receipt](/assets/images/3-morphological-operations/receipt.png)

Say we're given this receipt and we're tasked to read it automatically.

Let's perform the following cleaning pipeline:

1. Read the image as grayscale.
2. Blur and binarize the image.
3. Perform morphological operations to remove the lines.

First, let's read this image and convert it to grayscale:

```python
from skimage import io

img = io.imread('receipt.png', as_gray=True)
io.imshow(img);
```

![receipt read](/assets/images/3-morphological-operations/receipt-read.png)

Now, let's blur the image. We blur the image so as to remove any major noise in the image.

```python
from skimage.filters import gaussian

blurred = gaussian(img, 1)
io.imshow(blurred);
```

![blur image](/assets/images/3-morphological-operations/blur.png)

We then binarize it so that the morphological operations will be much more intuitive when we perform them.

```python
# Binarize using threshold
binarized = (blurred < 0.40)*1
io.imshow(binarized, cmap='gray');
```

![binarized](/assets/images/3-morphological-operations/binarized.png)

Now it becomes apparent of what we should do in this image. Since we want to remove the lines in the image, we can perform an opening operation with a structuring element of a vertical pixel to effectively erode the horizontal lines. Let's do this in the next cell:

```python
from skimage.morphology import opening

# Perform opening using a vertical structure element
cleaned = opening(binarized, np.ones((4, 1)))
io.imshow(cleaned, cmap='gray')
```
![no lines](/assets/images/3-morphological-operations/no-lines.png)

Aaaand, viola! No more lines! Well, of course our procedure is far from perfect. There are noticeable artifacts such as the letters becoming more fat. We can also explore to use other filters to effectively perform the cleaning task. One suggestion would be to perform edge detection then curve fitting so that the resulting letters are not as thick as this one.

Nonetheless we demonstrated here the role of morphological operation in cleaning an image, great! 💯

## Conclusion

In this blog, we demonstrated and introduced several morphological operations and showed the effect of the structuring element in the procedure. We also performed a cleaning operation for an OCR task were we removed the lines found in a receipt.

Take note that we only scratched the surface here, we will notice how vital these morphological operations are when we do end-to-end image processing projects and the crucial role they play in our model's accuracy and success.

That's it, friends! Thank you for your time reading this blog! Till next time! Peace out!
