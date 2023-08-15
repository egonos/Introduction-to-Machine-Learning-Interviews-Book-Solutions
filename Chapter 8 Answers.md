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

Autoencoders, clustering algorithmsi isolation forests

**4.**

* i. Why do we use feature selection?

-> Reduced required memory (due to eliminating unnecesary predictors)

-> Increased training speed (less computational requirement)

-> Increased model performance (ml models utilize only relevant features during training)

-> Reduced overfitting (Irrelevant features might causes th ml model to overfit)


