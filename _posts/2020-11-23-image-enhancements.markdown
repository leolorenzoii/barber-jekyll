---
layout: post
title: Image Enhancements
date: 2020-11-23
description: What do you do when a picture doesn't come out right? Do you scrap it? Do you take another shot? Or do you use <i>python</i> on it?
image: /assets/images/2-image-enhancements/postcard.png
author: Leo Lorenzo II
tags: 
  - skimage
  - image enhancements
---

What do you do when a picture doesn't come out right? Do you scrap it? Do you take another shot? Or do you use *python* on it?

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post, we'll get around the beaten path, and enhance the image using some techniques using `python`!

## What is Image Enhancement?

Image enhancement is similar to image editing, which "encompasses the process of altering images, whether they are digital photographs, traditional photo-chemical photographs or illustrations" (taken from [Image editing](https://en.m.wikipedia.org/wiki/Image_editing)). So basically, anything that involves making an image better looking, would count as image enhancement.

Here, we will show how we can use `python` to improve the quality of our image via three techniques: (1) White-balancing, (2) Histogram Manipulation, and (3) Fourier Transforms.

### White-balancing

**White balancing** or **color-balancing** as they call it in some books, is the global adjustment of the colors to render specific colors correctly. We employ this method each time we want to emphasize or correct certain parts on our image. Let's use this photo which I took a three years ago. I snap away without much regard to the lighting or quality of the image. Hence, we have a poorly lighted image:

![poorly lighted image](/assets/images/2-image-enhancements/orig.jpg)

Let's load this image to python using `skimage`.

```python
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

img = io.imread('orig.jpg')
resized = resize(img, (1500, 1200))
io.imshow(resized);
```

![loaded-image](/assets/images/2-image-enhancements/print-orig.png)

As you can see, this image is dull. We just have a white sky white some sad grey elements below. Let's try to use white-balancing to liven up the colors in this image.

There are many ways to perform white balancing, we can opt to just divide the values by some arbitrary value, such as the maximum value of the intensity of each color channel, or some corresponding percentile. But for me, the more intuitive way is to pick a patch on the image that we think should be white. This is the **ground-truth** algorithm.

In our photo, we can notice that we have a store sign in the lower-right half of the image, that should be white. Hence, we select that image and impose that it should have a white color when rendered in our image.

```python
patch = resized[900:970, 940:960, :]
io.imshow(patch)
```

![patch image](/assets/images/2-image-enhancements/patch.png)

Notice that the patch looks really gray, when we know that this should be color white. What would happen if we impose that this patch should have a color white in our image? Let's see.

```python
# Initialize figure and axes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Perform white balancing on the image
white_balanced = (resized*1.0 / patch.max(axis=(0, 1))).clip(0, 1)

# Show results
axes[0].imshow(resized)
axes[1].imshow(white_balanced)

# Set labels
axes[0].set_title('Before', fontsize=14, weight='bold')
axes[1].set_title('After', fontsize=14, weight='bold')

# Off axis
axes[0].axis('off')
axes[1].axis('off');
```

![White Balanced Image](/assets/images/2-image-enhancements/white-balanced.png)

Notice that the colors in the second picture are more livelier. The reds are brighter, the car looks shinier, everything looks more real. However, there's a drawback, if we observe, the texture of the clouds now could not be well differentiated. This is because, prior to white balancing, the clouds were already white. Essentially what we did is we lowered the ceiling for the white pixel, thus after white balancing, the clouds and the signboard that we chose as reference patch are now rendered at the same white color.

But then again, if want we want is to emphasize the colors of the objects on the lower half of our image, this trade-off is well justified.

### Histogram Equalization

Now, let's use another equally powerful algorithm to fix our image. This is the histogram equalization algorithm. To make things easier to explain, let's get a gray scaled image of what we used before, but focusing on the lower half of the image.

```python
from skimage.color import rgb2gray

grayed = rgb2gray(resized)
io.imshow(grayed);
```

![grayed image](/assets/images/2-image-enhancements/grayed.png)

Notice again, that in our image, elements are quite hard to distinguish. First, let's look at the histogram of the intensity of the gray values in our image.

```python
# Import seaborn for histogram plotting
import seaborn as sns

# Generate histogram plot
sns.histplot(grayed.flatten());
```

![histogram](/assets/images/2-image-enhancements/histogram.png)

Notice that our histogram is somewhat focused and concentrated on one side. For the elements of the image to be more distinguishable, it is preferred that the intensities are more spread out. This is exactly what **histogram manipulation** do, we re-distribute the intensity of the pixels such that it will span the whole range of values of the intensities. Let's look how this is performed in action:

```python
# Get frequency of the gray values
freq, bins = cumulative_distribution(img_as_ubyte(grayed))
target_bins = np.arange(255)
target_freq = np.linspace(0, 1, len(target_bins))

# Plot the actual cdf and the target cdf
plt.step(bins, freq, c='b', label='actual cdf')
plt.plot(target_bins, target_freq, c='r', label='target cdf')

# Set axis labels
plt.xlabel("Gray-level Intensity")
plt.ylabel("Cumulative Fraction of Pixels")

# Draw legend
plt.legend();
```

![cdf](/assets/images/2-image-enhancements/cdf.png)

The above figure shows the actual cumulative distribution of the gray level intensity versus our target. Again, as pointed during the histogram plot, the gray level values are somewhat concentrated on the lower values. What we want is a more flatter and even distribution such as the `target cdf` indicated here. So how will we do it?

One way is to use `numpy`'s `interp` to essentially create a mapping between the current intensity and target intensity, we demonstrate this in the cell below:

```python
# Generate mapping
mapping = np.interp(freq, target_freq, target_bins)

# Get histogram equalized image
equalized = img_as_ubyte(mapping[img_as_ubyte(grayed)].astype('uint8'))
```

Here, our the numpy interpolation array `mapping` creates a sort of translation mechanism to convert the current intensity values to the target values. As such when we plot the equalized image, we get:

```python
io.imshow(equalized);
```

![equaluized image](/assets/images/2-image-enhancements/equalized.png)

Ah yes, now there is light! We can clearly distinguish now every elements on the picture. Notice that what the equalization did is to separate the intensity values so that they are more easily distinguishable. As such, values near the maximum would be given an intensity value near 1.0. This may not be what we desire sometimes if we want the color of the image to be nearer to the true value. But if what we want is to be able to distinguish each elements on an image more clearly, histogram equalization is the way to go!

### Fourier Transform

The eagle has landed! I repeat, the eagle has landed!

Now imagine this, your partner has landed on the moon, and you have a photograph to capture that one big moment, that one small step but a giant leap for mankind! And after coming back on Eart, you noticed that the sensors of your camera was f*** up. What will you do?

All hope is not lost because with the use of some magic python image processing techniques, we can still possibly salvage that picture with the help of fourier transforms!

Let's look at our test image for this demonstration:

```python
img = io.imread('moon_image.png')
io.imshow(img);
```

![moon image](/assets/images/2-image-enhancements/moon-img.png)

Notice that our image has some well defined artifact. A repeating pattern that looks like a stamp. Now, this kinds of artifact can be produced based on the cause of equipment malfunction that we experience. Nonetheless, we can still sort of fix this by using filters from the frequency domain.

Let's look at the image's fourier transform to see what we can do:

```python
# Get moon image and fft iamge
moon = rgb2gray(io.imread('moon_image.png'))
moon_fft = np.fft.fftshift(np.fft.fft2(moon))

# Show image and its fourier transform
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(moon, cmap='gray')
ax[1].imshow(np.log(abs(moon_fft)), cmap='gray');
```

![fft image](/assets/images/2-image-enhancements/fft-img.png)

On the right we see the fourier transformed image. The bright spots that we see in the image comprises of elements that are repeating at a certain frequency in our original image. This means that theoretically, if we filter those bright spots, we can retrieve an image with the repeating elements filtered. In other words, we can clean the image. Let's try that out.

Let's use a mask that looks something like this:

```python
io.imshow(mask)
```

![mask image](/assets/images/2-image-enhancements/mask.png)

Note that I made this on my own using paint! Hahahaa. Of course you could do something like this using python, but there's just too much hard coding involve that my stomach couldn't take it! Haha.

Anyway's let's try to filter out the elements using this mask:

```python
moon_fft2 = moon_fft.copy()
moon_fft2[mask] = 1
```

Let's see how the new image looks like as compared from before:

```python
# Initialize axes
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# Show original image and the cleaned fourier transform image
axes[0].imshow(moon, cmap='gray')
axes[1].imshow(abs(np.fft.ifft2(moon_fft2)), cmap='gray')

# Remove axes then set the labels
for ax in axes:
    ax.axis('off')
axes[0].set_title("Before", fontsize=14, weight='bold')
axes[1].set_title("After", fontsize=14, weight='bold')
```

![mask image](/assets/images/2-image-enhancements/bf-fft.png)

Notice what happened. The distinct artifact has been removed. But the image looks like someone smudge an eraser on our image! Hahaha. This is also an artifact of our process. But let's appreciate the significant cleaning that happened after we filtered some elements on the Fourier transformed image.

I don't know about you, but the first time I did this, and when it worked, it felt like magic to me! Haha.

## Conclusion

In this blog post, we showed three different techniques for image enhancements namely: white balancing, histogram equalization, and fourier transforms. Each has their own strength and weaknesses, and they may be appropriately used depending on the scenario and type of cleaning that you want. Nonetheless, we have to be wary of the possible side effects of each techniques.

That's it, friends! Watch out here for more exciting blogs about image processing, data science, and life! Till next time!