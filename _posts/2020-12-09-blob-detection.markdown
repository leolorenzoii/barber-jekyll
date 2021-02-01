---
layout: post
title: Blob Detection
date: 2020-12-09
description: A blob is a drop of thick liquid or viscous substance. It is a spot of a color. A thing that you would easily notice in an image or staining your shirt.
image: /assets/images/4-blob-detection/rbc-bw.png
author: Leo Lorenzo II
tags: 
  - skimage
  - blob detection
---

A blob is a drop of thick liquid or viscous substance. It is a spot of a color. A thing that you would easily notice in an image or staining your shirt.

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post, we won't talk about blobs in the context of those slimy monsters that we usually slay when we're low level in our favorite RPG games nor will we talk about those guilty pleasure foods that keep on staining your shit, we'll talk about blobs in the context of image processing!

## What is blob detection?

In the context of image processing, **blob detection methods** aim at identifying regions that differ in properties such as brightness or color. Hence, we can say that a blob is a region in image wherein some properties are constant or approximately constant at all.

This means that in a mathematical context, when we want to detect *blobs*, what we really just need to look at is whether at a certain region, there is significant change in some properties that we are inspecting, whether it be color, area, or texture.

## What are methods for blob detection?

Automatic blob detection methods such as **Laplacian of Gaussian (LoG)**, **Difference of Gaussian (DoG)**, and **Determinant of Hessian (DoH)** leverage this fact and basically are all gradient based. However in this blog, we will demonstrate something simpler but I find using very often in our image processing exercises. Of course, we don't discount the fact that the three aforementioned methods are useless, they all have their own place and appropriate scenarios as to where they may be used effectively. But basing from experience, nothing is more easier, intuitive, and all-purpose to use when performing blob detection or image segmentation as the **Connected Components Labelling** method.

## What is the connected components labelling method?

One drawback of the three gaussian methods is that they look for circular objects in the image. Connected components can detect irregular shapes and sizes but needs a proper cleaning preprocessing for it to work effectively. 

With this in mind, let's then ask, how does connected component work? 

Well you guess it, as the name implies it determines which pixels are connected to each other. Now, this may be more complicated than it seems, because in the first place, how do we say that a pixel is connected with one another?

We can go about this in two ways (at least for 2D images):

We can say a pixel is connected to each other if another pixel lies directly in the north, south, east, or west direction of a given pixel:

![four connectivity](/assets/images/4-blob-detection/four-connect.png)

This is what we call a **4-connected neighbor**. As you can see this is more restrictive that usual since it has to be directly above, below, on the right, or on the left. We can also define neighbors as lying in the diagonal of a pixel of interest, in which case we call this **8-conneted neighbor** (see image below).

![eight connectivity](/assets/images/4-blob-detection/eight-connect.png)

This choice in type of connectivity can mean whether a certain region would merge or be splitted using the connected components labelling algorithm. Let's demonstrate how we perform this using `skimage`.

## Connected component labelling in `skimage`

Let's first define libraries that we will use:

```python
import numpy as np
import matplotlib.pyplot as plt
```

Then let's define our dummy data:

```python
img = np.zeros((6, 17))
img[1, 2::4] = 1
img[1, 3::4] = 1
img[2, 1:9] = 1
img[2, -6:-2] = 1
img[3, 3:7] = 1
img[3, 10:14] = 1
img[4, 2:6] = 1
img[4, 9:12] = 1
img[4, -3:-1] = 1
```

How does our data look like? Let's see.

```python
![dummy image](/assets/images/4-blob-detection/img.png)
```

To perform the *connected components labelling algorithm*, we just use the `label` function in the `skimage.measure` library:

```python
from skimage.measure import label
```

Let's take the case first where we consider neighbors of four-connectivity:

```python
plt.imshow(label(img, connectivity=1));
```

![four connectivity demo](/assets/images/4-blob-detection/img-4.png)

The crucial part here is the region found on the lower right. Notice that in our four-connectivity setting (`connectivity=1`), that pixel was disconnected with the other pixels. However, if we set `connectivity=2`, which implies we are defining neighbors as eight-connected neighbors, the pixels on the right of the image should be consider as one group:

```python
plt.imshow(label(img, connectivity=2));
```

![four connectivity demo](/assets/images/4-blob-detection/img-8.png)

In the case of eight-connected neighbors, we only have two groups instead of three. Which is what we expected based on our intuition of four-connected and eight-connected neighbors.

Now in practice, how can this simple concept be very useful? Let's look at the case of a classic example of blob detection, i.e., red blood cell counting:

## Blob detection applications

Say we are given this microscope image of red blood cells:

![red blood cell](/assets/images/4-blob-detection/rbc.JPG)

We can do this manually of course, but it would be cumbersome and tiring. So instead, let's perform connected components to distinguish the cell with the background and automatically count the number of cells.

Our cleaning process would be to first convert the image to grayscale, blur the image to remove noise, then binarize the image using a threshold (found through trial and error). After binarization, it would be straightforward to count the red blood cells via connected components labelling method.

Here are some libraries that we will use:

```python
from skimage import io
from skimage.filters import gaussian
from skimage.measure import label
plt.rcParams["figure.figsize"] = (16, 8)
```

Let's perform these steps accordingly, first let's read the image in python as grayscale:

```python
img = io.imread('rbc.JPG', as_gray=True)
plt.imshow(img);
```

![red blood cell](/assets/images/4-blob-detection/rbc-python.png)

Afterwards, we blur the image using a `gaussian` filter:

```python
blurred = gaussian(img, 1)
plt.imshow(blurred);
```

![red blood cell](/assets/images/4-blob-detection/blurred.png)

Then binarize it using a threshold we found through trial and error:

```python
binarized = blurred < 0.75
plt.imshow(binarized);
```

![red blood cell](/assets/images/4-blob-detection/binarized.png)

Finally, using connected components labelling, we can easily count the number of red blood cells in the image.

```python
labelled = label(binarized)
print(f"Number of red blood cells: {max(labelled.flatten())}")
```
    Number of red blood cells: 375

Here, our automatic count of red blood cell is `375`. Notice that there are some challenges in ensuring that this estimation is accurate. First, the lighting in the image plays a large role in determining the optimal threshold value to use during binarization. Second, some of the cells overlap with each other, hence, some may be connected only once. On hindsight, this might cause our estimation to be smaller than the actual number of the red blood cells.

To further improve our result, one suggestion might be to use a different threshold in some parts of the image that are overexposed. Another possible approach would be to perform edge detection first in order to separate overlapping cells.

## Conclusion

In this blog, we discussed how connected component works and how powerful it can be when used in applications such as red blood cell counting. Connected component labelling method is surely one of the tools in image processing that would be useful in a wide variety of scenarios specially coupled with `regionprops` which we will discuss in future blogs. Be sure to stay tuned for that! Haha!

**That's it, friends!** Thank you for your time reading this blog! Till next time! Peace out!