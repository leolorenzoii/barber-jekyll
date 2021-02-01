---
layout: post
title: RG Chromaticity Space
date: 2021-01-06
description: Segmenting colors is hard! There's too much parameters to tune!
image: /assets/images/6-non-parametric-segmentation/card.png
author: Leo Lorenzo II
tags: 
  - skimage
  - image segmentation
---

Segmenting colors is hard! There's too much parameters to tune!


**Hello, hello! Friends, classmates, and my idols in life!** In our previous two blog posts we demonstrated how to segment images based on their color by utilizing their differences in the color spaces whether RGB or HSV. Notice that it is quite cumbersome to create segmentation masks using these color spaces as you have to investigate and inspect multiple parameters in order to achieve your objective of segmenting a certain color.

Is there any improvements that we can make regarding segmentation? Fear not, my friends. For today, you're in for a treat! We will use introduce another color image segmentation method: **Non-parameteric Segmentation** using the **RG-Chromaticity Space** to effectively segment based on color space but will less parameters! This makes our life easier but of course there's always some caveat here.

## What is the RG-Chromaticity space?

RG-Chromaticity space is similar to the color spaces that we know. The main difference is that in this space, there is no intensity information. Meaning, there is less parameters to worry about but it gives us less information regarding the image. Of course, this is specially useful for the case where we deal only with the color of the image, since the RG-Chromaticity gives us that information directly.

So how do we achieve this?

## How to convert between RGB and the RG Chromaticity?

The main idea here is that for the RG Chromaticity space, a color is represented by the proportion of red, green, and blue in the color rather than by the intensity of the each. Meaning for this to work, we get the proportional intensity of each channel depending on the total intensity values for the given pixel.

The first step therefore to convert between $(R, G, B)$ values to $(r, g)$ values is to get the normalized color space via the equations:

$$
r = \frac{R}{R + G + B}
$$

$$
g = \frac{G}{R + G + B}
$$

$$
b = \frac{B}{R + G + B}
$$

Afterwards, we notice that we can represent any color just by using the $r$ and $g$, information. Here, notice that it is trivial to solve for the $b$ value given the $r$ and $g$ value since it is implied that $r + g + b = 1$. This is encapsulated by the following figure:

![chromaticity](/assets/images/6-non-parametric-segmentation/chromaticity.png)

Here notice that if we go r=1, g=0, we get the red color. Likewise, for r=0, g=1, we get the green color. For g=0, and r=0, we get the blue color. Everything else in between can also be represented by specifying just the r and g values. Amazing! Isn't it?

## How to use RG Chromaticity Space in Color Segmentation

Now, knowing the RG Chromaticity Space, how do we use this in our color image segmentation problem? Well, one thing we can do is inspect a patch, look at where in the RG Chromaticity space is that patch located, then look for pixels with rg values similar to our specified patch. This is what is known as **non-parameteric segmentation**. 

## Non-parameteric Segmentation

Let's implement this *non-parameteric segmentation* using python.

Let's look at the bean bag image that we used in the previous blog post and implement this method when we segment the orange bean bag. Let's again define our helper function `draw_bbox` to help us select the relevant patches.

```python
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

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

Now, let's select a patch from the orange bean bag as our reference patch.

```python
# Draw image
img = io.imread('bags.png')

# Select patch
box_o = (200, 250, 100, 150)

# Plot results
plt.imshow(img)
ax = plt.gca()
draw_box(box_o, ax)
```

![chromaticity](/assets/images/6-non-parametric-segmentation/orange-bag.png)

Now, let's get the RG chromaticity values of the patch and the image

```python
# Select patch
patch = img[box_o[0]:box_o[1], box_o[2]:box_o[3], :]

# Get RG chromaticity values for patch and image
patch_r = patch[:, :, 0]*1.0/patch.sum(axis=2)
patch_g = patch[:, :, 1]*1.0/patch.sum(axis=2)
img_r = img[:, :, 0]*1.0/img.sum(axis=2)
img_g = img[:, :, 1]*1.0/img.sum(axis=2)
```

To inspect where in our RG chromaticity space our patch is, we just need to plot its values:

```python
plt.figure(figsize=(5,5))
plt.hist2d(r.flatten(), g.flatten(), bins=16,cmap='binary')
plt.xlim(0,1)
plt.ylim(0,1);
```

![chromaticity](/assets/images/6-non-parametric-segmentation/orange.png)

Notice that the RG-chromaticity values lie somewhere between `0.40` to `0.80` in r, and `0.20` to `0.30` in g. Of course we can create a big rectangular mask using those values to segment orange colors, but we would get a lot of noise if we do that. The more elegant way is to use backprojection to get the ROI in our image.

To do this, first we get the histogram values of the rg chromaticity of the patch:

```python
# Get histogram values
H, x, y = np.histogram2d(patch_r.flatten(), patch_g.flatten())

# Set mesh of histogram values
mesh = np.dstack(np.meshgrid(x, y))[:10, :10, :]
mesh = mesh[H > 0]
```

Here, the `mesh` variable contains all the combinations of R and G that have non zero values in the plot shown awhile ago. This means these are the RG chromaticity values of the patch. The next step is to take the difference of these rg chromaticity, with the rg chromaticity values of the image pixels. We do this via the following cell:

```python
# RG Chromaticity values of the image
img_rg = np.dstack((img_r, img_g))

# Take difference between rg values from patch and image
hist2pix = (np.abs(img_rg.reshape(img_rg.shape[0],
                                  img_rg.shape[1], 1, 2) - mesh)
              .mean(axis=3).min(axis=2))
```

Finally, we set a suitable threshold to use. This threshold just tells us how far from the patch values do we want the selected pixels are. Meaning, this threshold controls how lenient or mask is, during the selection of pixels.

```python
# Set threshold
thresh = 2.8e-2

# Plot result
plt.imshow(hist2pix < thresh);
```

![chromaticity](/assets/images/6-non-parametric-segmentation/segmented.png)

Aaand VOILA! Notice that here in our segmentation process, we did not need to inspect the histogram values of each of the respective colors and the only threshold that we set here is our tolerance or leniency with respect to matching the rg chromaticity values found in our selected patch.

## Conclusion

In this blog post, we show how we can perform color image segmentation using the RG Chromaticity space. We found that here, using hte non-paramteric segmentation, we only needed to set one hyperparameter threshold. Although this makes the tuning process easier, of course there is less flexibility and all hinges on the rg-chromaticity values that was selected when we obtained the patch.

This will surely be one of the methods I'd go back to when we need to segment the image according to the color because of its intuitive and light approach. However, for more complex segmentation, where I need to look at the saturation or intensity values, I'd probably look at the HSV space instead.

**That's it, friends!** Thank you for your time reading this blog! Till next time! Peace out!

## Bonus!

As a bonus, here is a function that we can use to perform the non-parameteric segmentation just by specifying the image array and the bounding box of the patch:

```python
def non_param_mask(img, bbox, thresh=1e-2):
    """
    Return non parameteric segmentation mask given image and bbox
    
    Parameters
    ----------
    img : numpy array
        Numpy array of the image of interest
    bbox : numpy array
        Bounding box of the patch in the image
        (xmin, xmax, ymin, ymax)
    thresh : float, defaul=1e-2
        Threshold to use during segmentation
    
    Returns
    -------
    mask : numpy array
        Boleean mask of the segmented image
    """
    # Get patch
    patch = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    
    # Get RG values of image and patch
    patch_R, patch_G = get_RG(patch)
    img_R, img_G = get_RG(img)
    
    # Get histogram values
    H, x, y = np.histogram2d(patch_R.flatten(), patch_G.flatten())
    
    # Set mesh of histogram values
    mesh = np.dstack(np.meshgrid(x, y))[:10, :10, :]
    mesh = mesh[H > 0]
    
    # Stack RG values of image
    img_RG = np.dstack((img_R, img_G))
    
    # Get pixel values from histogram values
    hist2pix = (np.abs(img_RG.reshape(img_RG.shape[0],
                                       img_RG.shape[1], 1, 2) - mesh)
                  .mean(axis=3).min(axis=2))
    
    # Return mask given threshold
    return (hist2pix < thresh).reshape((img.shape[0], img.shape[1], 1))
```