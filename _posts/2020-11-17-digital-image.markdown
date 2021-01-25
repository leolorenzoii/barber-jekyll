---
layout: post
title: "The Digital Image"
date: 2020-11-17
description: 
image: /assets/images/placeholder-9.jpg
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

To demonstrate, of course, we would use something that could do those operations efficiently, right?

Enter, `python` the ultimate programming language that can do almost anything (almost!).

Let's tackle the first question in everybody's mind. How do you open an image using `python`? Well, in photoshop, paint, or whatever, what we do is open an image using the graphic interface of that application, right? For `python`, we do the same, but we codify it, of course!

### Reading an image using `Python`

We'll do a simple operation of opening an image, then performing some operations such as `resizing`, `quantizing`, and converting it to a `grayscale`. These operations are trivial but here let's dig deep on what happens when we do these operations and demonstrate them using python.

Let's use the image of *Lena*, the lovely woman you saw in the preview of this blog post.

```python
import numpy as np
from skimage import io
img = io.imread('lena.png')
```

The above cell reads the `png` image stored in our directory as a variable `img`. But what does this `img` look like? Let's print this in our notebook to see.

```python
img
```
  array([[[225, 137, 127],
          [224, 137, 127],
          [227, 134, 119],
          ...,
          [227, 141, 128],
          [232, 150, 124],
          [213, 120, 104]],

As you can see, the image of *Lena* is represented as numbers. Wherein each number corresponds to the intensity values of the **3** color channels, **RGB (Red-Green-Blue)**. There are other ways to represent this three channels which include the **HSV (Hue-Saturation-Value)** and the **CMYK (Cyan-Magenta-Yellow and Key)**. We'll save the discussion of these channel for another day, because our objective for this post is doing some stuff on this image.

Without further adieu, let's resize this image of Lena.

### Re-scaling an image using python