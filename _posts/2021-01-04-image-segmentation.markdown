---
layout: post
title: Image Segmentation
date: 2021-01-04
description: How do you distinguish colors? How do you separate the red, the green, the blue? How about other colors?
image: /assets/images/5-image-segmentation/bags-card.png
author: Leo Lorenzo II
tags: 
  - skimage
  - image segmentation
---

How do you distinguish colors? How do you separate the red, the green, the blue? How about other colors?


**Hello, hello! Friends, classmates, and my idols in life!** In our previous two blog posts where we demonstrated **blob detection** and **morphological operations**, we have shown the *binarization* technique, wherein we separate foreground and background images using a certain threshold. In today's blog we push this concept further by segmenting images based on their colors, i.e. by discussing the concept of **Color Image Segmentation**.

## What is color image segmentation?

In computer vision or image processing, the idea of **image segmentation** is to partition a digital image into several regions or segments, depending on a recognizable property. This can be as basic as the binarization of images which we performed in the previous blog posts, or as complex as separating the organs or tissues in the body given a certain image.

With this in mind, **color image segmentation**, therefore, can be defined as segmenting parts on an image depending on the color of particular elements in the image.

Here, we will demonstrate how to perform color image segmentation through our basic knowledge of the color spaces. Specifically, we will treat the intensity values independently as described by the **RGB Color Space** and the **HSV Color Space**.

To do this demonstration let's consider an image of beanbags which was given to us by our image processing professor:

![bags](/assets/images/5-image-segmentation/bags.png)

As you can see, not only do these bean bags look cozy af but they consist of different colors: blue, yellow, orange, and green. We will try to segment this image by leveraging the fact that in the RGB and HSV color space, each of these bags have different and distinct values.

## Color image segmentation using RGB space

In essence, what we're going to do is not too different to the binarization techniques that we have done in the previous blog posts. However, when we did binarization, we only considered one intensity. Here, we will consider multiple intensity values and even other derivatives to achieve our objective.

To demonstrate this let's first read the image using python:

```python
from skimage import io
import matplotlib.pyplot as plt

img = io.imread('bags.png')
plt.imshow(img);
```

![bags](/assets/images/5-image-segmentation/bags-python.png)

Let's say we want to segment the blue bag from the other bag, our first step here is to look at its intensity values for each channel of the image:

```python
# Set labels
channels = ['Red', 'Green', 'Blue']

# Get intensity values per channel
img_r = (img[:, :, 0]/255).clip(0, 1)
img_g = (img[:, :, 1]/255).clip(0, 1)
img_b = (img[:, :, 2]/255).clip(0, 1)

# Plot intensity values on image
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
for ax, ch, label  in zip(axes, [img_r, img_g, img_b], channels):
    ax.imshow(ch, cmap='gray')
    ax.set_title(label, fontsize=14, weight='bold')
```

![bags](/assets/images/5-image-segmentation/bags-rgb.png)

Here, notice that the intensity values for the blue bag is low for the red channel, high for the blue channel. We can then leverage this fact by creating a mask that gets only the pixels that have low values for the red channel and high values for the blue channel.

Here, we can perform a trial and error method, but that would be cumbersome as we have three different channels to mix and match. Thus, we use a sort of automated way to determine the threhold for each channel, i.e., we use the `otsu` method for thresholding.

Through `otsu` method, the threshold is automatically determined by minimizing the intra-class intensity variance, or maximizing the inter-class variance. Which basically means that it will select the intensity value per channel that would greatly separate the foreground and the background based on their variances.

The `otsu` method can easily be implemented using `skimage`'s function: `threshold_otsu`

```python
from skimage.filters import threshold_otsu
```

Now, let's create our mask for the blue bag.

```python
# Blue bag has low values of red, and green intensities, but high blue channel
r_mask = img_r < threshold_otsu(img_r)
g_mask = img_g < threshold_otsu(img_g)
b_mask = img_b > threshold_otsu(img_b)
```

Here we leverage the fact that the blue bean bag has high intensity values for the blue channel, low intensity values for the red channel, and relatively low values for the green channel.

Let's look at what this mask would look like when put in together:

```python
# Show resulting mask
plt.imshow((r_mask) * (b_mask) * (g_mask))
```

![bags](/assets/images/5-image-segmentation/blue-bag.png)

Voila! We somehow segmented the blue image here automatically. But there's some sort of noise that is included in our mask. Can we improve it?

One approach is to get the pure blue intensities by subtracting it with the gray intensity values of the image. Afterwards, we can use `otsu` method again to perform image segmentation. This approach was suggested by one of our classmates, *Vee*. Credits to him for this approach!

```python
# Get blue intensities
blue = img[:, :, 2] - rgb2gray(img)*255
blue2 = np.where(blue > 0, blue, 0)

# Get threshold based on otsu method
thresh = threshold_otsu(blue2)

# Segment the image
plt.imshow(blue2.astype(int) > thresh)
```

![bags](/assets/images/5-image-segmentation/blue-bag2.png)

Notice here that the segmented blue bean bag is much more cleaner as compared to our first approach.

### Color image segmentation using HSV color space

Now, even though segmenting images using the RGB color space is intuitive. This has its own limitations, for example, segmenting orange using the RGB color space would be challenging since we don't expect it to peak at one of our channels. To solve this challenge, one approach is to use the HSV color space instead.

Let's look at how the intensity values appear at the HSV color space:

```python
# Get hsv color space
img_hsv = rgb2hsv(img)

# Initialize figure and labels
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
labels = ['Hue', 'Saturation', 'Value']

# Iterate through all channel
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.imshow(img_hsv[:, :, i], cmap='gray')
    ax.set_title(label, fontsize=14, weight='bold')
```

![bags](/assets/images/5-image-segmentation/bags-hsv.png)

Notice here that the `hue` value separates the four colors smoothly and the saturation values of all the four bean bag are very high since their colors are well defined.

Let's try to segment now the orange, green, and yellow bean bag using the HSV color space. First, let's look at how the `Hue` values of each bean bag varies. We will use the helper function `draw_bbox` to help us in our patch selection:

```python
def draw_box(box_coords, ax, edgecolor='r', lw=1.5):
    """Draw bounding box coordinates on given axes"""
    # Get corner coordinate
    patch = (box_coords[2], box_coords[0])
    
    # Get height and width
    height = box_coords[1] - box_coords[0]
    width = box_coords[3] - box_coords[2]
    
    # Plot patch on axes
    ax.add_patch(Rectangle(patch, width, height, edgecolor=edgecolor, lw=lw,
                           facecolor='none'))
```

Let's select a patch sample for each of the bean bag:

```python
# Draw hue of image
plt.imshow(img_hsv[:, :, 0], cmap='gray')
ax = plt.gca()

# Select patches
box_o = (200, 250, 100, 150)
box_g = (200, 250, 300, 350)
box_y = (60, 100, 350, 400)

# Draw boxes
draw_box(box_o, ax, 'orange')
draw_box(box_g, ax, 'green')
draw_box(box_y, ax, 'yellow')
```

![bags](/assets/images/5-image-segmentation/bag-patches.png)

Now, let's look at the histogram plot of each color patches in terms of Hue and Saturation values. This will aid us in our creation of masks per color of the bean bag.

```python
# Initialize figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Draw the histogram plot of each
sns.distplot(img_hsv[box_o[0]:box_o[1], box_o[2]:box_o[3], 0], ax=axes[0])
sns.distplot(img_hsv[box_g[0]:box_g[1], box_g[2]:box_g[3], 0], ax=axes[1])
sns.distplot(img_hsv[box_y[0]:box_y[1], box_y[2]:box_y[3], 0], ax=axes[2])

# Draw titles
titles = ['Orange', 'Green', 'Yellow']
for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=14)
plt.suptitle('Hue Values', fontsize=16, weight='bold');
```

![bags](/assets/images/5-image-segmentation/bags-hue.png)

Here we notice that for the orange bean bag, hue values vary from `0.00` to `0.07`, while for the green bean bag, hue values may vary from `0.20` to `0.60`, finally for the yellow bean bag, hue values vary from `0.08` to `0.20`.

```python
# Initialize figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Draw saturation values
sns.distplot(img_hsv[box_o[0]:box_o[1], box_o[2]:box_o[3], 1], ax=axes[0])
sns.distplot(img_hsv[box_g[0]:box_g[1], box_g[2]:box_g[3], 1], ax=axes[1])
sns.distplot(img_hsv[box_y[0]:box_y[1], box_y[2]:box_y[3], 1], ax=axes[2])

# Draw titles
titles = ['Orange', 'Green', 'Yellow']
for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=14)
plt.suptitle('Saturation', weight='bold', fontsize=16)
```

![bags](/assets/images/5-image-segmentation/bags-saturation.png)

In terms of the saturation, we notice that for the orange bean bag, the minimum value is `0.50`, for the green bag, the minimum value is `0.24`, while for the yellow bag, the minimum value is `0.70`.

We include all of these observations in our creation of the mask to get the following result:

```python
# Initialize figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.set_title(titles[i])

# Orange mask
lower_mask = img_hsv[:,:,0] > 0.00
upper_mask = img_hsv[:,:,0] < 0.07
sat_mask = img_hsv[:, :, 1] > 0.50
mask_o = upper_mask*lower_mask*sat_mask

# Green mask
lower_mask = img_hsv[:,:,0] > 0.20
upper_mask = img_hsv[:,:,0] < 0.60
sat_mask = img_hsv[:, :, 1] > 0.24
mask_g = upper_mask*lower_mask*sat_mask

# Yellow mask
lower_mask = img_hsv[:,:,0] > 0.08
upper_mask = img_hsv[:,:,0] < 0.20
sat_mask = img_hsv[:, :, 1] > 0.70
value_mask = img_hsv
mask_y = upper_mask*lower_mask*sat_mask

# Plot results
axes[0].imshow(mask_o)
axes[1].imshow(mask_g)
axes[2].imshow(mask_y);
```

![bags](/assets/images/5-image-segmentation/bags-ogy.png)

Aaaand, VOILA!! Notice that we cleanly separated the orange bean bag, green bean bag, and yellow bean bag using HSV space. Amazing!

## Conclusion

In this blog post, we showed how we can extend our binarization and thresholding technique to colored images by independently considering the intensity values of the image per channel. We created mask depending on the observed behaviour of our target object. Then combined each mask using a sort of an `and` operator by multiplying each mask. Through this, we effectively segment objects based on their color! Very cool indeed!