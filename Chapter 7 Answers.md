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

********

**7.** What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

If we think the loss function as a path in the parameter space, local minima represents a minimum point relative to its nearby points. However this point is not the global minimum.

In the saddle point the gradinet of f is 0 Saddle points seem to be worse. (ref: https://or.stackexchange.com/questions/7778/quality-of-solutions-from-saddle-points-vs-local-minimums)

**8.**

* i. What are the differences between parameters and hyperparameters?

Parameters are the coefficients of the model. Consider: WTX = y_hat. W is the parameters of the model to estimate the outcome.

Hyperparameters on the other hand the *settings* of the model. They are determined previous to training and based on these values, the style of training changes. Ex: Solver: liblinear

* ii. Why is hyperparameter tuning important?

The best settings for each case changes. Therefore, to get most out of a model it should be tuned.

* iii. Explain algorithm for tuning hyperparameters.

Let me explain the GridSearch because it is the most straightforward tuning mechanism on ML world. In GridSearch we iteratively try the parameter comibnations on the model. In the end we select the best hyperparameter set and arrange the settings accordingly.

**9.** 

* i. What makes a classification problem different from a regression problem?

A classificaiton problem can be handled based on monitoring the probabilities and expected values whereas a regression problem is more straightforward. The model determines the coefficients and computes the outcome. Based on the error, it updates its parameters in every iteration.

* ii. Can a classification problem be turned into a regression problem and vice versa?

Actually by using probabilities and changing the loss function, we turn a classification problem into a regression problem. 

Converting regression problem to a classification probelm is also possible. Instead of trying to estimate a number the model can estimate whether estimated number is in group 1, 2, 3 etc.

**10.**

* i. What’s the difference between parametric methods and non-parametric methods? Give an example of each method.

Parametric methods builds the model on top of assumptions. (Possibly the base assumption is data is normally distributed.)
After making these assumptions it finds parameters like mean and standard deviation to explain the patterns lie in the data. Ex: Linear Regression

Nonparametric method makes no assumption on data. Due to this reason, it requires more computational power and data. Ex: KNN

* ii. When should we use one and when should we use the other? 

If we have less time, data or computational resources or we are confident enough that the data is alligned with our assumptions, we should use parametric methods.


**11.** Why does ensembling independently trained models generally improve performance?

Because the expected value of combining predictions is higher compared to using only one. Consider this example 

A model 70% accuracy -> Expected_value(X) = 0.7*1 = 0.7

An ensemble model containing 70% accurate 5 submodels.

Expected_value(X) = 1 - (5_3)(0.3)(0.3)(0.3)(0.7)(0.7) - (5_4)(0.3)(0.3)(0.3)(0.3)(0.7) - (0.3)**5 = 0.86

**12.** Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?

Consider gradient descent. When we use L2 norm for regularization, change in gradient will be exponential therefore the final value won't be 0. The graident vanishes before that happens.

On the other hand, L1 norm is a subgradient meaning all the values not being 0 treated as the same (same gradient) Eventually the final value becomes zero.

![Regularization](Images/Regularization%20Comparison.png)


**13.** Why does an ML model’s performance degrade in production?



**14.** What problems might we run into when deploying large machine learning models?

**15.** Your model performs really well on the test set but poorly in production.
* i. What are your hypotheses about the causes?
* ii. How do you validate whether your hypotheses are correct?
[M] Imagine your hypotheses about the causes are correct. What would you do to address them?