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

**8.** How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

We can employ a handful of statsitical tests to determine whether two set of samples belng to the same distribution:

* Mann-Whitney U Test: This is a non-parametric test used to determine whether two sample groups belong to the same distribution. To compute this test effectively, the variables should be ordered because the calculations are made based on the order of the values.  (ref: https://acikders.ankara.edu.tr/pluginfile.php/30763/mod_resource/content/0/10_Mann%20Whitney%20U%20Testi.pdf)

*t-test: This test is used to determine whether the means of two different statistical distributions are significantly different. (ref: https://acikders.ankara.edu.tr/pluginfile.php/169671/mod_resource/content/0/12_T%20TEST%C4%B0.pdf)

* Chi-Square Test: This test is suitable for evaluating whether two sets of categorical data come from the same distribution.


**9.** How do you know you’ve collected enough samples to train your ML model?

The best way to see the answer of this question is to plot learning curves of training and validation data. If training and cross validation loss are similar given data, increasing the number of instances won't improve the perforance. Conversely, if there is a gap between those two, the model would probably benefit from increased data size:

! [CV](Images/Cross%20Validation%20Loss.png)


**10.** How to determine outliers in your data samples? What to do with them?

1. If the data is somewhat normal then IQR based approaches work really well. Similarly we can use box plots or violin plots for the same purpose.

2. If the data is not normal, scatter plots can be really handy. To visualize the data we can preprocess the data using dimension reduction techniques such as PCA as well.

3. We can also use AutoEncoders to detect outliers. The compression and expansion processes show us the outliers in our data if there any.

4. Clustering algorithms such as KMeans (if the data is spherical) or DBSCAN (if the data is not spherical) can be useful.

**11.** Sample duplication

* i. When should you remove duplicate training samples? When shouldn’t you?

Sometimes duplicates occur naturally in the data. For example, when working with multiple tweets, "Fantastic!" would probably occur more than once. In these cases, we should never drop the duplicates. On the other hand, duplicates can occur because of an error during data collection, processing, or transfer. In these cases, we should get rid of these duplicates.

* ii. What happens if we accidentally duplicate every data point in your train set or in your test set?

1. If it won't cause a problem, I would drop the duplicates.

2. Alternatively, I can take the row form of the data and apply the same procedures until the recent error is resolved.

**12.** Missing data

* i.  In your dataset, two out of 20 variables have more than 30% missing values. What would you do?

1. If the feature is not important, I would drop it.
2. If it's important, then I'll try to fill the missing values depending on the case. Sometimes the sample statistics are sufficient. In other cases, we may need to use more advanced missing value handling techniques. In my experience, `XGBImputer` works really well.

* ii. How might techniques that handle missing data make selection bias worse? How do you handle this bias?

Using simpler approaches can be helpful. As we employ more advanced techniques, the model selectively picks the samples and uses them to fill the missing values. This may cause overfitting to occur. On the other hand, simpler approaches such as using sample statistics do not have this problem.

**13.** Why is randomization important when designing experiments (experimental design)?

Because of our limitations, we cannot conduct experiments on the entire population. To solve this problem, we use randomized samples. Fortunately, when we pick our samples randomly, we can gain insights about the entire population using the Central Limit Theorem (CLT).


**14.** Class imbalance.

* i. How would class imbalance affect your model?

Class imbalance is a difficult concept to deal with. A model trained on an imbalanced dataset is expected to be biased towards the majority class, resulting in poor generalization performance.

* ii. Why is it hard for ML models to perform well on data with class imbalance?

Since the model is biased, its predictions are likely to lean towards the majority class. This works well for the training data but not for real-case evaluations.

* iii. Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?

To improve the model, I would lower the decision threshold, increasing recall at the expense of precision.

**15.** Training data leakage.

* i. Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?

Oversampling may provide more instances but it doesn't mean better learning. In fact, since the positive classes are mostly the same, the model tends to memorize these duplicates rather than learning general patterns. That's why it performs worse in production.

* ii. You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?

ChatGPT:

> Temporal Patterns: If the data has temporal patterns (e.g., more spam comments during weekends), then a random split would distribute these patterns across both the training and test sets. This could make the model artificially good at recognizing these patterns, as they would be present in the test set. A better approach would be to use a time-based split, where you train on older data and test on newer data.

>Duplicate or Near-Duplicate Comments: If there are duplicate or near-duplicate spam comments, a random split could place some of these in the training set and others in the test set. The model would then easily recognize these in the test set, inflating its performance metrics.

> User Behavior: If multiple comments from the same user are present, and that user tends to produce either mostly spam or mostly non-spam comments, then a random split could distribute comments from this user across both sets. This would make it easier for the model to classify comments from this user in the test set, again inflating performance metrics.

**16.** How does data sparsity affect your models?

Data sparsity increases the dimension of the data without serving further information. For the distance based models (like KNN, KMeans etc.) it results in *curse of dimensionality*. On the other hand, even my data is resilient for the curse of dimensionality (tree models), the increased dimensions take more computational power and time.

**17.** Feature leakage

* i. What are some causes of feature leakage?

1. Ordinal features and ordered labels: If I have an id column and ordered labels, the model automatically develops a decision criteria as:

y = 1 if id < 150 (let's say)

y = 0 otherwise

2. Mulitcolinearity: If any of two data is not **linearly independent**, this may cause the model to develop wrong learning patterns. Luckily we can monitor this via several methods. Personally I use pearson correlations (continuous data) and chi-square of independence test (categorical data)


* ii. Why does normalization help prevent feature leakage?

NO IDEA. According to the ChatGPT:

> Normalization helps prevent feature leakage by ensuring that the scale of the variables in your model is determined solely by the training set and not by the test set. When you normalize based on the entire dataset, the mean and variance used for scaling include information from the test set, thus leaking information into the training set. This is problematic because it can give an optimistic bias to the evaluation metrics of the model when tested on the same test set, making the model look better than it actually is. By normalizing using only the training data, you ensure that the model is completely ignorant of the future data in the test set, offering a more realistic evaluation of its performance.

My opinion: This is not the answer we're looking for. I couln't find the answer of this question on the internet also.

* iii. How do you detect feature leakage?

Answered in (i.)

**18.** Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?

In this case, the fact that validation and test data consists a minority part of the whole data, the model evaluation metrics might skew a lot. If somehow the distributions of these two differs even in a small scale this quickly reflects the performance scores. To solve this problem we can use cross validation.

**19.** You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?

If the categorical features have low cardinality, I would directly encode them using OHE. However if the otherwise is true depending on the data type we can use other encoding methods. Here are my most frequently used ones:

Categorical data (not text): Mean encoding

Categorical data (text): Word embeddings

**20.** Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?

I think both training and test error would increase if I use distance based model. As I have mentioned previously, the curse of dimensionality can overwhelm the ML models a lot. The distances grow exponentially while we are increasing dimensions. Here is an illustration:


 ![Curse](Images/Curse%20of%20Dimensionality.png)


Independent from this, increasing features may result in overfitting.
