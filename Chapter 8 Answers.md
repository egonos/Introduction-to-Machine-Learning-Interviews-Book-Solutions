# 8.1.2 Questions

**1.** What are the basic assumptions to be made for linear regression?

* The data is normally distibuted.

Ideally,

* Predictors are not correlated.
* Each data point is equally significant.

**2.** What happens if we don’t apply feature scaling to logistic regression?

If we use gradient descent,

Assume the vector space is in 2-D. If we does not scale thew predictors and there is a scale difference between those two, the contours looks like an amplified ellipse. The direction of the movement (opposite direction to the steepest descent) won't be towards the minima. That's why we observe some zig-zags. On the other hand, if we scale the features, the contour lines look like a circle. In that case, the direction of the movement always show the global minima.


If we use analytical methods,

The coefficients assigned to each predictor become too different from each other. Therefore, the interpretability decreases. Independent from this, the model won't be trained optimally when the scales are not matching (remember regularizers use distances).

**3.** What are the algorithms you’d use when developing the prototype of a fraud detection model?

Autoencoders, clustering algorithms, isolation forests

**4.**

* i. Why do we use feature selection?

-> Reduced required memory (due to eliminating unnecesary predictors)

-> Increased training speed (less computational requirement)

-> Increased model performance (ml models utilize only relevant features during training)

-> Reduced overfitting (Irrelevant features might causes th ml model to overfit)

* ii. What are some of the algorithms for feature selection? Pros and cons of each.

-> Tree algorithms: Resistant to scale differences. Able to capture **non-linear** realtionships. Prone to overfitting.

-> Linear models: Simple. Prone to underfitting. LASSO can be used for feature selection (makes the coefficients of the irrelevant features 0)

-> Variance inflation factor (to decrease predictor relevancies). Only considers linear relationships.

**5.** 

* i. How would you choose the value of k?

-> **Elbow method:** We have to get get the most bang for our buck i.e. when distortion descrease becomes marginal then we need to stop to increase k.

-> **Sector requirements:** Production availability, customer response etc.

* ii. If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?

By comparing the clustering results with true labels.

* iii. How would you do it if the labels aren’t known?

-> Dimension reduction (if necessary) and visualization. In this way I also can decide which clustering algorithm is appropriate for this spesific case.

-> Look at the distortion measure (we can use silhouette score as well). If increasing k still has a potential to deduce the distortion **significantly**, then I increase it. The opposite is true while decreasing the number of k.

* iv. Given the following dataset, can you predict how K-means clustering works on it? Explain.

No it's not going to work. DBSCAN would perform lot better. K-means will not perform the best in cyclic datasets. The distance between data points is minimized when the data has a spherical shape. Therefore in a non-sperical geometry, K-means can not perform well.

**6.**

* i. How would you choose the value of k?

Empirically. We have to find a balance between overfitting (small k) and underfitting (large k).

* ii. What happens when you increase or decrease the value of k?

**Decreasing k:** Model becomes more sensitive to any neighbor of the data point in consideration. This is a good thing up to a point. We wan't to prioritize the nearest neighbors while assigning the labels. However, this can quickly lead to overfitting. Considering more neighbors can make the model more robust.

**Increasing k:** Model considers more neighbors while assigning the label of the data in consideration. This make the mdoel more resiliant to noises (little changes with no importance)