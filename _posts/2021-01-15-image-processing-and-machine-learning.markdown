---
layout: post
title: Image Processing annd Machine Learning
date: 2021-01-15
description: When two things go together so well, that means that there is synergy between the two.
image: /assets/images/8-ip-ml/card.png
author: Leo Lorenzo II
tags: 
  - skimage
  - machine learning
---

When two things go together so well, that means that there is synergy between the two.

In the past few blogs we have discussed several image processing techniques to essentially: segment, transform, and characterize the elements in an image. Let's take it one step further and use our learnings to solve a machine learning problem.

**Hello, hello! Friends, classmates, and my idols in life!** In today's blog post we look how we can use the image processing techniques we have discussed in the past few blogs and solve a machine learning problem with a relevant use case: *leaf classification*.

We will discuss here an end to end machine learning pipeline which involves pre-processing of the image, feature extraction and derivation, and finally machine learning.

## The Core Problem

Before anything else, let's first discuss the core problem that we will tackle in today's blog post. Essentially we have several images of different classes of leaves (A, B, C, D, and E) with noticeable differences in their characteristic such as area, shape, and size.

![leaves](/assets/images/8-ip-ml/leaves.png)

The leaf images were obtained by scanning the several leaf samples ontop of a bond paper through a scanner. Thus, we can see some artifacts such as the texture of the bond paper, as well as some leaves being too close together during the scanning process.

The main challenge here is to find which features would easily discriminate the leaves from one another. Here, most notable is the area, since both plant A and plant B have noticeably large area. Afterwards, plants C, D, and E would be harder to distinguish. We need to define some sort of derived paramters to effectively differentiate those leaves from one another.

## The Methodology

All in all we have 27 scanned photos, where each photo contains several samples of leaf. Thus, the preprocessing pipeline should be able to segment each leaf in each image. Afterwards, we can then use `skimage`'s `regionprops` function to extract features such as area, bounding box area, eccentricity, etc. We also derive some features such as perimeter over equivalent distance to distinguish hard to differentiate leaves. Then finally, we use traditional machine learning models to perform our leaf classification.

## Data Preprocessing

The preprocessing step is crucial for our problem. We need to effectively segment each leaves from one another to be able to extract the relevant features that would distinguish them later on. As such we performed a spotless preprocessing pipeline to perform this task. Thankfully, I have my teammates to help on this one, we designate cleaning each leaf samples accordingly with a coherent preprocessing pipeline. In a nutshell our preprocessing pipeline looks like this:

1. Binarize the image
2. Clean holes through `area closing`
3. Separate leaves that are overlapping
4. Label the image
5. Get the region properties of each group

Our preprocessing pipeline code looks something like this:

```python
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import area_closing


# Read image
img = io.imread(filepath, as_gray=True)

# Binarized image using otsu
bin_img = img < threshold_otsu(img)

# Clean image by filling in holes
cleaned_img = area_closing(bin_img, 32)

# Label image
label_img = label(cleaned_img)

# Get region props
props = regionprops(label_img)
```

Of course, there would be some special cases such as separating too close leaves which required additional cleaning step such as the following below:

```python
# Set exception for plant A 5
if (plant_label == 'A') and (img_num == 5):
    # Add edge filter for overlapping leaves
    edges = sobel(img)
    m = mask(img, (500, 600, 320, 350)).reshape(img.shape[0],
                                                img.shape[1])
    edge_mask = m*closing(edges > 0.05, selem=np.ones((20, 1)))
    cleaned_img = ~edge_mask*cleaned_img
```

Here we used the `sobel` edge detection, emphasize those edges, then trimmed the detected edges on the original image so that connected leaves would be separated.

## Feature Extraction

After preprocessing we can now have the default features as provided by the regionprops. These are namely:

1. area
2. bbox_area
3. convex_area
4. eccentricity
5. equivalent_diameter
6. extent
7. major_axis_length
8. minor_axis_length
9. perimeter
10. solidity

However, in order to distinguish more effectively the leaves from one another, we opted to define derived features namely:

11. length
12. width
13. p/ed
14. a/p
15. ca/p

`length` and `width` refers to the bounding box length and width, which we observe to be a determining factor for long and short leaves. `p/ed` refers to the ratio of perimeter and the equivalend distance, this feature gives us a scale invariant description of the leave and its perimeter, which we hope would be able to distinguish leaves C, D, and E. `a/p` and `ca/p` refer to area over perimeter and convex area over perimeter respectively. Both of these features attempt to describe the shape of the leaf but with size taken into account.

## Machine Learning

We have our features, we have our labels. The next step of course is to feed everything that we've done to a machine learning algorithm. There's a lot to choose from, but intuitively, since we only got a handful of samples, we expect that linear models would tend to generalize better (as oppose to tree based models) in our case since we have too few examples. Nonetheless, the machine learning models that we trained and tested for this problem are the following:

1. `kNN`
2. `Logistic (L1)`
3. `Logistc (L2)`
4. `SVM (L1)`
5. `SVM (L2)`
6. `Decision Tree`
7. `Random Forest Classifier`
8. `Gradient Boosing Classifier`
9. `AdaBoost Decision Tree Classifier`

In order to perform training and testing, we used an "auto-ML" code (which came from a code that I augmented during our machine learning class). This code trains several hyperparmeters for each model then testing them a total of `10` times each.

The result of our training and testing is as follows:

![machine learning](/assets/images/8-ip-ml/ml.PNG)

Here, notice that our intuition is echoed by the result. The best model is the SVM (L1) with an accuracy of **94.68%**. This is quite high than what I expected given that even I would have trouble differentiating some of the leaves in the dataset. The top predictor here says that it is `solidity`, but the caveat here is that the top predictor here only says which of the coefficient is higher. Thus, the more appropriate description is that the model is most sensitive from the `solidity` feature.

To get the importance of each feature, looking at the **permutation importance** plot is more appropriate:

![permutation importance](/assets/images/8-ip-ml/pm.png)

Here, it is fleshed out that the `area` is indeed the most important parameter to distinguish the leaves from the dataset. Which is fairly intuitive given that leaves A, and B have substantial areas as compared to the other leaves. The next interesting parameter here is the `a/p`. Which reveals that this derived parameter is crucial to discriminate the other leaves specifically C, D, and E. The perimeter ratio is crucial, since as we expected, leaves with ragged edges would have lesser `a/p` as compared to smoother leaves.

## Conclusion

In this blog post, we showed how we created an end to end machine learning pipeline with image processing essentially at the center piece of feature extraction. Here, we emphasized the importance of finding derived features. The feature `a/p` was crucial to discriminate the difficult to distinguish leaves C, D, and E.

This problem goes to show that when it comes especially to image processing and machine learning, the creativity in terms of finding and defining features would be vital to create accurate models and results.

**That's it, friends!** Thank you for your time reading this blog! Till next time! Peace out!

