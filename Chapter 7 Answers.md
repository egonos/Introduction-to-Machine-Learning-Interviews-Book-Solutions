# 7.1 Basics

**1.** Explain supervised, unsupervised, weakly supervised semi-supervised, and active learning.

**Supervised Learning:** Given the set of data and labels model tries to capture the patterns.

**Unsupervised Learning:** Given the set of data, model tries to cluster them (due to grouping or dimension reduction purposes)

**Weakly Supervised:** Weakly supervised learning is the same as supervised learning. The only difference is that the trained labels are not finely generated (low quality).

-> "Contains a dog" instead of "dog"

-> We use heuristics for the labeling.

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

# 7.2 Sampling and creating training data

1. If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?

(6_2)*(4_1) = 60 ways

2.  What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?

**Sampling with replacement:** On each trial the probability of each sample to be picked is 1/n. This means one sample caould be picked more than one time.

**Sampling without replacement:** On each trial, the number of samples decreases. That's why the chance of a sample in the population to be picked increases (1/n, 1/(n-1)...).

Lets say we have a large population which is hard to sample. Luckily we have a sample group. However since we have only one sample group, we can not apply Cental Limit Theorem directly. By bootstraping, we can multiply the sample groups we have and by using CLT, without collecting more samples we can gain some insights about the population.

On the other hand, if the sampling is done fpr pther purposes, then adopting sampling without replacement would be a better choice. For example when deciding the winner of the lottery, if we use sampling with replacement then one lucky person could win more than one prize which is meaningless. 

**3.** Recall Accept Reject Sampling:

In Accept Reject sampling we are trying to sample from a probability distribution f(x) which is hard to sample from. We use a g(x) ~ N(μ,σ) for that purpose. Since it is Gaussian, we can easily pick samples from g(x). Then accept-reject part comes into play. Based on the probability f(x)/(M*g(x)); M ∈ N we accept or reject the samples that we have picked from g(x). Since M scales the distribution g(x), the scaled version covers f(x).

The problem arises when f(x) is too complex. The M we need to use becomes too big resulting accepting probability f(x)/(M*g(x)) to be too small. Because of this problem we use MCMC. Different from Accept Reject sampling, in MCMC we utilize the last sample to get the next sample (Markov Chain Assumption: The next state is dependent on the last state). First we find the stationary distribution of Markov Chain. Then assume the accept reject probability as p(x'/x) where p(x'/x) is the transition probability from x' to x. By doing this we are not dependent of M anymore so small accept probabilities won't become an issue.

**4.** If you need to sample from high-dimensional data, which sampling method would you choose?

I would prefer Gibbs Sampling. Gibbs sampling is a MCMC method for multidimensional cases. To use Gibbs sampling, the following two conditions should be satisfied:

* p(x1|x2,x3,x4...xn), p(x2|x1,x3,x4...xn) are easy to sample from.

* p(x1,x2.....xn) is hard to sample from.

The algorithm is pretty simple. Consider 2D p(x,y)

1. Start with (x_0,y_0).
2. x_1 ~ p(x_1|y_0); y_1 ~ p(y_1|x_1)
3. We obtained ours second sample: (x_1,y_1)

...

**5.** Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.

Consider (xi,Ti) ∈ L where xi are the features, Ti are the labels and L is a large universe. F(x,y) is our compatibility function. By using F, our goal is to explain the pattern(s) behind all (xi,Ti) pairs. When L is too large, computing F(x,y) can be really compututationally expensive. When then happens, we try to pick a sample group called Candidates Ci ⊂ L and compute function F only in Ci domain. Generally Ci consists of the real targets Ti and randomly selected samples belonging to other classes Si

Ci = Ti U Si

An example:

Word2Vec

Ti => Word pairs within the window 

Si => Randomly selected word pairs not in the window

F(x,y) => log(P(y|x)/Q(y|x))



**6.** Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.

* i. How would you sample 100K comments to label?

I would start with Random Sampling however since the data is likely to be imbalanced i.e. comments violating website's rule is significantly less than the rest, this approach may not be the best.

If the results allign with my expectations, I would proceed with stratified sampling. In this way, I would be more confident about my sample group because I know that it contains both of the classes.

Alternatively, I can play with the sampling probabilities of the samples. For example assigning a higher probabibility for the minority group can contribute significantly to our sampling process.

Another alternative could be Quota Sampling. Let's say I want 50K for both classes. Until I reach the limit, I'll continue to sample.


* ii.  Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?

I would pick at least 30 samples for each stratum because, for the Central Limit Theorem (CLT) to be valid (or to make healthy statistical inferences), we need at least 30 samples.

**7.** Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?

There is a mismatch between the concepts that have been talked. My friend talks about all the articles published. Conversely, I'm talking about the artickles that I have been translated. My selections are biased towards the publisher (me). That's why the facts are not alligning with each other.


There is a mismatch between the concepts that have been talked about. My friend talks about all the articles published. Conversely, I'm talking about the articles that have been translated. My selections are biased towards the publisher (me). That's why the facts are not aligning with each other.

How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

We can employ a handful of statsitical tests to determine whether two set of samples belng to the same distribution:

* Mann-Whitney U Test: This is a non-parametric test used to determine whether two sample groups belong to the same distribution. To compute this test effectively the variables should be ordered because the calculations are made based on the order of the values. (ref: https://acikders.ankara.edu.tr/pluginfile.php/30763/mod_resource/content/0/10_Mann%20Whitney%20U%20Testi.pdf)

* t-test: To see whether the difference of the mean belonging to two distinct statistical distributions. (ref: https://acikders.ankara.edu.tr/pluginfile.php/169671/mod_resource/content/0/12_T%20TEST%C4%B0.pdf)

* Chi-Square Test: To work on the same topic but for categorical data.


