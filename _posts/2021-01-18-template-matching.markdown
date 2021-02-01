---
layout: post
title: Template Matching
date: 2021-01-18
description: When we're searching for something, be it an object, a person, or something else. It pays to have a template in our mind that would determine the characteristics of what we are searching for.
image: /assets/images/9-template-matching/card.png
author: Leo Lorenzo II
tags: 
  - skimage
  - template matching
---

When we're searching for something, be it an object, a person, or something else. It pays to have a template in our mind that would determine the characteristics of what we are searching for.

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post we look how we can use template matching to find and essentially count objects that look similar to a given template.


## What is template matching?

In digital image processing, template matching is finding small parts of an image that will match a given template image. It has various applications in fields such as quality control, or navigating machine robots, and a way to detect edges in an images.

In this blog, we will demonstrate this by using an image of aircraft carrier, then detect the aircrafts that can be found from this image. We will use `skimage`'s `match_template` function.

First, let's discuss how does this template matching work?

## How does template matching work?

In essence `match_template` performs template matching via fast, normalized cross_correlation. What this does is essentially use the input template as a kernel then perform cross correlation on the given bigger image. Values that would match the pixels of the template onto the image would give higher values, and values lower would be tagged as those not matching with the given template.

## Template matching demonstration

As a demonstration let's perform this on an aircraft carrier image. Here, we of course have several aircrafts resting on top of an aircraft carrier:

```python
from skimage import io
import matplotlib.pyplot as plt

# Read aircraft carrier image
img = io.imread('aircraft_carrier.jpg', as_gray=True)
plt.imshow(img, cmap='gray');
```

![aircraft carrier](/assets/images/9-template-matching/img.png)

We then select a particular image of an aircraft and use that as our template:

```python
# Select template
template = img[648:744,775:838]
plt.imshow(template, cmap='gray');
```

![template](/assets/images/9-template-matching/template.png)

Let's perform the `match_template` function and see what it returns as its result:

```python
from skimage.feature import match_template

# Perform match template
result = match_template(img, template)
plt.imshow(result, cmap='gray');
```

![match template](/assets/images/9-template-matching/match_template.png)

What this result shows you essentially is the result of the cross correlation operation. The range of the values of the result would vary from `-1` to `1`, with `1` being the perfect match between the template and the corresponding region on the image. 

```python
print(f"Result shape: {result.shape}, Image shape: {img.shape}")
```
    Result shape: (1283, 2038), Image shape: (1378, 2100)

Notice here that the shape of the result is smaller than the shape of the original image. This is because when we perform cross correlation given a template, the dimensions would reduce according to dimensions of the template.

To retrieve which among the parts of the image would be considered as a "match" with our template, we need to use `peak_local_max` function of `skimage`. This allows us to define a threshold through which we can retrieve locations at which the cross correlation result of `match_template` yields a result greater than the threshold that we chose.

```python
from skimage.feature import peak_local_max

# Show aircraft carrier image
plt.imshow(img, cmap='gray')

# Set template width and height
template_width, template_height = template.shape

# Select matches based on peak_local_max
for x, y in peak_local_max(result, threshold_abs=0.40):
    rect = plt.Rectangle((y, x), template_height, template_width, color='y', 
                         fc='none')
    plt.gca().add_patch(rect);
```

![match template](/assets/images/9-template-matching/res1.png)

Notice that increasing the threshold here makes our algorithm more stringent, thus lesser match will occur.

```python
from skimage.feature import peak_local_max

# Show aircraft carrier image
plt.imshow(img, cmap='gray')

# Set template width and height
template_width, template_height = template.shape

# Select matches based on peak_local_max
for x, y in peak_local_max(result, threshold_abs=0.80):
    rect = plt.Rectangle((y, x), template_height, template_width, color='y', 
                         fc='none')
    plt.gca().add_patch(rect);
```

![match template](/assets/images/9-template-matching/res2.png)

On the other hand, using a smaller threshold would make our algorithm more lenient, leading to more matches.

## Limitations

Our template matching algorithm performs great in our demonstration. The threshold value specification also provides some degree of flexibility that allows our algorithm to be lenient or stricter as we please. However, we do note that since the core concept at which template matching operates is cross correlation, this means that if we use a template that differs on the resolution of what we expect in the image, we would get an unfavorable result.

This means that if our template is too enlarge or too distorted (rotated or disfigured in some way), the cross correlation operation would fail.

## Conclusion

In this blog post, we demonstrated how to use the `match_template` function of `skimage`. Additionally by using the `peak_local_max` function, we were able to control the leniency of the cross correlation algorithm in determining which locations in the bigger image matches the given template.

I personally would like to apply template matching to problems relating to following or predicting trajectory of an objects, since it would be a cool application of this concept.

**That's it, friends!** Thank you for your time reading this blog! Till next time! Peace out!