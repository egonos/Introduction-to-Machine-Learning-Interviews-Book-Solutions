# 7.1 Basics

1. Explain supervised, unsupervised, weakly supervised semi-supervised, and active learning.

**Supervised Learning:** Given the set of data and labels model tries to capture the patterns.

**Unsupervised Learning:** Given the set of data, model tries to cluster them (due to grouping or dimension reduction purposes)

**Weakly Supervised:** .......

**Semi-supervised:** There are lots of data but small potion of it also has labels. Given the data and the labels, model tries to capture the patterns and apply what it learned to all the data.

**Active-Learning:** ......................

2. 

* i. What’s the risk in empirical risk minimization?

We have set of functions F resulting in labels y in supervised learning. Now, we want to represent F with f. To monitor sufficiency of f on representing F, we use loss functions:

L(f(x),y)

where f(x) is attempt to represent F given predictor set x. The risk is the attempt itself. By using f to represent F, how much we can mimic the behaviour of F. If we make f too simple then it couldn't give the same results with F. On the other hand, if we make f too complex than f becomes too sensitive to noises, it becomes unstable and again can not mimic the behaviour of F.


* ii.  Why is it empirical?

We can not process the whole system X so we sample x (the observations of the system, makes really small percentage of the whole data) from X and make predictions accordingly. This what makes the risk empirical.

Assumption:

Expected_value(f(x)) = y

* iii. How do we minimize that risk?

Ensemble Methods (Trees)

Regularization

Cross Validation

Adopting simpler models if possible

 4. What are the conditions that allowed deep learning to gain popularity in the last decade?

-> Increased data size

-> Ability to record sensory data (hearth rate for instance)

-> Increased computational resources