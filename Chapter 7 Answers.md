# 7.1 Basics

**1.** Explain supervised, unsupervised, weakly supervised semi-supervised, and active learning.

**Supervised Learning:** Given the set of data and labels model tries to capture the patterns.

**Unsupervised Learning:** Given the set of data, model tries to cluster them (due to grouping or dimension reduction purposes)

**Weakly Supervised:** Weakly supervised learning is the same as supervised learning. The only difference is that the trained labels are not finely generated (low quality).

-> "Contains a dog" instead of "dog"

**Semi-supervised:** There are lots of data but small potion of it also has labels. Given the data and the labels, model tries to capture the patterns and apply what it learned to all the data.

**Active-Learning:** In active learning, model has a right to choose a subset of the data giving the best results. By employing this technique, a model performs well while training less. Here's the process:

* The model trains on a very small subset of labeled data.
* It goes to unlabeled data and picks the ones in which it has the least confidence (close to decision boundries).
* The model gives the selected data to an expert (the user in front of the computer)
* The user labels the data and gives it back to the model.

This cycle continues until the training ends.



**2.** 

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

**3.** Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?

**In my view,** the less dependent we are while converging the ground truth the better. Employing more staff should only be considered if necessary because this process exponentially increase the probability for us to make mistakes.

 **4.** What are the conditions that allowed deep learning to gain popularity in the last decade?

-> Increased data size

-> Ability to record sensory data (hearth rate for instance)

-> Increased computational resources

**5.** If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

Deep NN's are more expressive because it processes the data more firmly therefore the final predictors become the essence of the data.


**6.** The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?

NEED HELP

7. What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

If we think the loss function as a path in the parameter space, local minima represents a minimum point relative to its nearby points. However this point is not the global minimum.

In the saddle point the gradinet of f is 0 Saddle points seem to be worse. (ref: https://or.stackexchange.com/questions/7778/quality-of-solutions-from-saddle-points-vs-local-minimums)