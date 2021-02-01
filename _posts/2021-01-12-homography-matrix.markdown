---
layout: post
title: Homography Matrix
date: 2021-01-12
description: Hey, make sure that the camera is angled correctly! I don't want my double chin to be emphasized in the picture!
image: /assets/images/7-homography-matrix/card.png
author: Leo Lorenzo II
tags: 
  - skimage
  - homography matrix
---

Hey, make sure that the camera is angled correctly! I don't want my double chin to be emphasized in the picture!

When taking pictures, it is crucial that the camera angle is pointing parallel to the horizontal, else we would see some unwanted artifacts such as bloated heads or bloated foots.


**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post we look at the perspective or projections. Specifically, how can we map one planar projection on an image to another. In layman's lingo (or instagram lingo if we must), this just means that we can convert *any* planar projection to say a *flat lay* projection that we oh so love in instagram! Haha.

We will discuss here specifically the concept of *Homography* and *Homography Matrices* how we implement this on the use case of sports.

## What is homography?

Homography describes the relationship of any two planar projections of an image. It is represented by a matrix, known as, *homography matrix*, which can be a combination of rotaion, translation, scaling or skew operations.

In essence, homography matrices describe how one coordinate system will be translated to another coordinate system. Let's demonstrate the case where we translate, rotate, and scale the image:

**Translation**

![translate](/assets/images/7-homography-matrix/translate.png)

**Rotation**

![rotate](/assets/images/7-homography-matrix/rotate.png)

**Scale**

![scale](/assets/images/7-homography-matrix/scale.png)

Here, notice that specific parts of the matrix have their own specific effects. In general, homography combines all of those effects in a one whole matrix:

**Homography**

![homography](/assets/images/7-homography-matrix/homography.png)

Notice here that we have `8` different variables that we need to specify. This also means that if we were to perform homography projection or transformation, we need at least to specify at least 8 coordinates in order to have a well defined transformation. For cases were we only expect translation or rotation, we need fewer coordinates to perform the transformation.

## Homography in Action

Usually, we are interested in the problem, given an image, how can we tranform the image to another perspective. In this case, we try to perform a tranformation from one coordinate system to the other given known points.

Let's demonstrate this by transforming a still image of a basketball game then convert it to a some sort of flat-layed top-view image coordinate system:

```python
# Read images
still1 = io.imread('still1.png')
still2 = io.imread('still2.png')
court = io.imread('court.png')

# Show images
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels = ['Image 1', 'Image 2', 'Top View']
for i, img in enumerate([still1, still2, court]):
    axes[i].imshow(img)
    axes[i].set_title(labels[i], fontsize=14)
```

![images](/assets/images/7-homography-matrix/imgs.png)

Here, we have three perspective or three coordinates system: image 1 from `still1`, Image 2 from `still2`, and the top view image, our target coordinate system.

Our approach will be to select specific points in the court then translate it to the top view image. Let's deal with `still1` image first. The `5` noticeable points on the court is the perimeter box and the upper left edge corner of the court. Let us select those points:

```python
from matplotlib.patches import Circle
import seaborn as sns

# Initialize figure
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Show image
ax.imshow(court)
X = [6, 6, 6, 174, 172]
Y = [6, 183, 329, 183, 329]
dst = list(zip(X, Y))

# Designate points
for i, (x, y) in enumerate(zip(X, Y)):
    patch = Circle((x, y), 5, color=sns.color_palette('tab10')[i])
    ax.add_artist(patch)
```
![coordinate 1](/assets/images/7-homography-matrix/c1.png)

Let's get the corresponding locations on `still1` image:

```python
# Initialize figure
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Draw coordinates in this space
ax.imshow(still1)
X = [695, 470, 200, 1188, 1020]
Y = [395, 550, 720, 600, 780]
src_s1 = list(zip(X, Y))

for i, (x, y) in enumerate(zip(X, Y)):
    patch = Circle((x, y), 20, color=sns.color_palette('tab10')[i], zorder=20)
    ax.add_artist(patch)
```

![still 1](/assets/images/7-homography-matrix/s1.png)

Next, let's use the `transform` function of `skimage` to essentially generate the `homography` matrix.

```python
from skimage import transform
import numpy as np

# Set rouce and destination points
src = np.array(src_s1)
dst = np.array(dst)

# Generate transform matrix
tform = transform.estimate_transform('projective', src, dst)
tf_img = transform.warp(still1, tform.inverse)

# Plot the result
fig, ax = plt.subplots()
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')
ax.set_xlim(0, court.shape[1])
ax.set_ylim(court.shape[0], 0);
```

![transform 1](/assets/images/7-homography-matrix/t1.png)

Aaand, VOILA! The original image was transformed in the top view.

Now, let's look at the second still image. Here, since there are more elements on the court that we can specify, we add two more points, i.e. the edge of the half court lines.

```python
# Initialize figure
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.imshow(court)

# Select points on the court
X = [6, 6, 6, 174, 172, 447, 447]
Y = [6, 183, 329, 183, 329, 6, 505]
dst = list(zip(X, Y))

# Show those points on the image
for i, (x, y) in enumerate(zip(X, Y)):
    patch = Circle((x, y), 5, color=sns.color_palette('tab10')[i])
    ax.add_artist(patch)
```

![coordinate 2](/assets/images/7-homography-matrix/c2.png)

Marking those same points on the court image

```python
# Initialize figure
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.imshow(still2)

# Select points
X = [445, 340, 205, 680, 605, 1205, 1197]
Y = [470, 540, 630, 550, 640, 490, 790]
src_s2 = list(zip(X, Y))

# Draw points on the image
for i, (x, y) in enumerate(zip(X, Y)):
    patch = Circle((x, y), 10, color=sns.color_palette('tab10')[i], zorder=20)
    ax.add_artist(patch)
```

![still 2](/assets/images/7-homography-matrix/s2.png)

Now, let's perform the homography matrix transformation given the source and data points:

```python
# Set source and destination points
src = np.array(src_s2)
dst = np.array(dst)

# Generate transformation matrix
tform = transform.estimate_transform('projective', src, dst)
tf_img = transform.warp(still2, tform.inverse)

# Show results
fig, ax = plt.subplots()
ax.imshow(tf_img)
_ = ax.set_title('projective transformation')
ax.set_xlim(0, court.shape[1])
ax.set_ylim(court.shape[0], 0);
```

![transform 2](/assets/images/7-homography-matrix/t2.png)

Aaand, there it is! We transformed the original image based on the selected points. Notice that the room for improvement here is to select more points, since as we discussed earlier, to create an effective homography tranformation you need to specify at least 8 coordinate points. Nonetheless, what's amazing is the estimate transform function of `skimage` was still able to give a homography matrix despite the lack of several points. Very cool, indeed!

## Conclusion

In this blog post, we discussed the concept of homography and showed how different matrix tranformations correspond to operations or transformations of our image (or essentially our coordinate system). We demonstrated one use case of this in sports, i.e. transforming one view to another view to have a better idea of location of players. Homography matrix applications also extend to things like panoramic picture stitching, camera pose estimation, and perspective removal.

**That's it, friends!** Thank you for your time reading this blog! Till next time! Peace out!
