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

* iii. How does the value of k impact the bias and variance?

Increasing k increases bias (more error) but decreases variance (less sensitive). Decreasing k has an opposite impact on the model.

**7.**

* i. Compare the two.

-> Gaussian Mixture Model (GMM) utilizes Gaussian Distribution to cluster (probablilistic) the data points whereas K-Means utilizes distance metrics (deterministic).

-> GMM assumes the data is normally distributed. K-Means assumes the data is spherical.

-> GMM support covariances for vairety of geometrical shapes (spherical,diagonal,tied and full covariance).

-> GMM needs sufficient amount of data to form clusters whereas K-Means can work even in very small data.

* ii. When would you choose one over another?

-> If the data distibution is close (overlapping clusters), K-Means can be inferior to detect the boundaries. GMM is a better choice in this case. 

-> If computational resouces are limited or the clusters have well defined boundaries, use K-Means.

**8.**

* i. What are some of the fundamental differences between bagging and boosting algorithms?

**Boosting**

-> Sequential Learning (Each learner learns the error of the previous learner.)

-> Increases variance reduces bias

-> Requires relatrively weak learners

**Bagging**

-> Simultaneous learning

-> Reduces the variance (might increase bias)

-> Learners can be weak or strong

* ii. How are they used in deep learning?

-> Tensorflow supports tree algorithm - NN connection. We can use Gradient Boosted trees or Random Forest as the head of our NN structure.

-> Creatively we can build some bagging and boosting structures with multiple NN models (This could be computationally heavy).
 
**9.**

* i. Construct its adjacency matrix.

![Adjency](Images/Adjency%20Matrix.png)

* ii. How would this matrix change if the graph is now undirected?

![Adjency2](Images/Adjency%20Matrix2.png)

* iii. What can you say about the adjacency matrices of two isomorphic graphs?

Since the number of verticies, edges and connectivities are the same among these two the adjency matricies could be transformed to each other by reordering the rows or the columns.


**10** Imagine we build a user-item collaborative filtering system to recommend to each user items similar to the items they’ve bought before.

* i. You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?

User-item matrix is more powerful in recommendation but the memory requirement of this matrix is also can be really big especially if there are lots of customers. Item-item matrix on the othr hand is lot more memory friendly, however it is not customer spesific i.e. treats all the customers the same and this may not be optimal.

* ii. How would you handle a new user who hasn’t made any purchases in the past?

Trust the prior knowledge obtained form customers similar to the new user.

**11.**Is feature scaling necessary for kernel methods?

Yes. Consider SVM. It constructs the decision boundaries based on distances. Therefore on a disproportional space, it's decision boundary assignments wouldn't be accurate.


**12.** Naive Bayes classifier.

* i. How is Naive Bayes classifier naive?

The algorithm directly assumes that all the predictors are conditionally independent from each other.

P(y|x1,x2) = P(y|x1) * P(y|x2)

* ii. Let’s try to construct a Naive Bayes classifier to classify whether a tweet has a positive or negative sentiment. We have four training samples


Tweet	                              Label

This makes me so upset	              Negative

This puppy makes me happy	          Positive

Look at this happy hamster	          Positive

No hamsters allowed in my house	      Negative

According to your classifier, what's sentiment of the sentence `The hamster is upset with the puppy?`


![Bayes1](Images/Bayes1.png)
![Bayes2](Images/Bayes2.png)


**13.** Two popular algorithms for winning Kaggle solutions are Light GBM and XGBoost. They are both gradient boosting algorithms.

* i. What is gradient boosting?

Graident boosting is a sequential learning technique in which each weak learner learns and possible corrects the errors made by the previous tree. The equation,

F_n+1(x) = F_n(x) - eta * yhat_n(x)

The biggest difference of Gradient Boosting from AdaBoost is that weak learners of a GB model is more advanced (not decision stumps).

* ii. What problems is gradient boosting good for?

In my experience GB outperforms most of the other ml algorithms when data is tabular and the data is not too big. Luckily novadays, both tree algorithms support GPU usage and parallel computing. (I found **Dask** to be more user friendly)

**14.** SVM

* i. What’s linear separation? Why is it desirable when we use SVM?

It means the decision boundary is a line (2D) or a hyperplane (+2D). It is desirable because linear decision boundaries are more straightforward to compute and useful when there is not a sufficient amount of data (KNN lies on the opposite side, the more data the better, slower computation)

* ii. How well would vanilla SVM work on this dataset?

-> Really nicely. Even hard margins can handle this separation

-> Good enough. One outlier point is not sufficient for SVM to fail. Especially by using soft margins we can accomplish sweet results.

-> Not good. SVM overwhelms a lot because the clusters are overlapping and there are outliers for each cluster.




# 8.2.1 Natural language processing


**1.** RNNs

* i. What’s the motivation for RNN?

Recurrent Neural Networks are degined to learn the sequential data like time series, text data etc. The math is:

![RNN](Images/RNN.png)


* ii. What’s the motivation for LSTM?

LSTM is designed for dealing with the vanishing gradient problem occuring in RNN. The math of VGB in RNN is:

![Derivative](Images/Derivative.png)

Vanihing gradient problem occurs as the model continues to learn due to stack of tanh'.

* iii. How would you do dropouts in an RNN?

WE can apply dropouts to input and output layer of RNN. However applying dropout to a hidden state is a bit tricky. Lets look at mathematically:

![DropoutRNN](Images/DropoutRNN.png)


2. What’s density estimation? Why do we say a language model is a density estimator?

Density estimation is an attempt for estimating a distribution using a parametric or non parametric methods.

**Parametric method:** We use a function to capture the patterns data points form. This is fast, but requires a prior geometric shape assumption (like sperical). 

-> Assume a shape

-> Find the best parameter combination to represent the data (like mean and variance)

**Nonparametric method:** We use data points themselves to create the density estimation (like in the KNN Classification). We don't have to have a geometric shape assumption but since we use data points themselves, we need sufficient amount of data. Moreover this process is slower and not as efficient.

-> Separate the data into small bins.

-> Combine the bins.


A language model prdicts the the outcome word probability distribution based on the data it gets.

I -> am

  -> belive

  -> do

I am -> happy

    -> tired

    -> a


**3.** Language models are often referred to as unsupervised learning, but some say its mechanism isn’t that different from supervised learning. What are your thoughts?

Based on the word set it gets, a language model tries to predict the next word. This is similar to the supervised learning where for each instance the model predicts a label. For the language models, each instance is the updated word set like the example above. On the other hand, it is understandable to assign the language models in unsupervised learning category because they don't require any labels to train with.

**4.** Word embeddings.

* i. Why do we need word embeddings?

Alternatively, we can use OHE for tokenization. The problem of this approach is that the vocabulary matrix can grow really easily on relatively long texts. Due its sizes this sparse matrix is really hard to work on. That's why we are using word embeddings.

* ii. What’s the difference between count-based and prediction-based word embeddings?

Prediction-based word embeddings include **context** (like Word2Vec) whereas count-based embeddings only considers the counts of a word (like TF-IDF).

* iii. Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?

A word may have more than one meaning. To solve this problem, we have to approach this problem more seriously. The assumption is a bit naive.

**5.** Given 5 documents:
 D1: The duck loves to eat the worm
 D2: The worm doesn’t like the early bird
 D3: The bird loves to get up early to get the worm
 D4: The bird gets the worm from the early duck
 D5: The duck and the birds are so different from each other but one thing they have in common is that they both get the worm

* i. Given a query Q: “The early bird gets the worm”, find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}. Are the top-ranked documents relevant to the query?

TF

* Bird: D2,D3,D4,D5
* Duck: D1,D4,D5
* Worm: D1,D2,D3,D4,D5
* Early: D2,D3,D4
* Get: D3,D4,D5
* Love: D1,D3

D3 is the most relevant because it contains early,worm,get,bird. Similarly D2 ranks the second. Yes D2 and D3 are relevant with Query.

* ii. Assume that document D5 goes on to tell more about the duck and the bird and mentions “bird” three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?

Since TF increases it's rank also increases. Although the intention is good, the fact that an irrelevant addition of a word to a document increases it's ranking, this could be somewhat misleading.

**6.** Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?

I would prefer n-grams because the Neural Language Models require lots of words to perform thier best. Moreover, starting with a simple model is always a good choice due to its low memory and computational requirements.




# 8.3 Training neural networks

**1.** When building a neural network, should you overfit or underfit it first?

I would underfit the first and increase the model complexity if necessary.

**2.** Write the vanilla gradient update.

Xt+1 = Xt - eta * ∇f(Xt)

**3.** Neural network in simple Numpy.
* i. Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.

Forward Pass:

    z1 = <w1,x>.............z1 = np.dot(w1,x)

    y1 = ReLU(z1)...........y1 = np.maximum(z1,0)

    z2 = <w2,y1>............z2 = np.dot(w2,y1)

    y2 = ReLU(z2) ..........y2 = np.maximum(z2,0)


Backward Pass:

     ∂L/∂wi = ∂L/∂y2* ∂y2/∂z2 * ∂z2/∂y1 * ∂y1/∂z1 * ∂z1/∂wi

     #MSE as loss fnc
     L = 0.5(y2-y)**2
     ∂L/∂y2 = y2-y

     #ReLU Derivative
     ∂y2/∂z2 = (z2>0).astype(float) # 1 in positive 0 in negative
     
     ∂z2/∂y1 = w1

    #ReLU Derivative
     ∂y1/∂z1 = (z1>0).astype(float)

     ∂z1/∂wi = xi

Lets combine all these on Python

  import numpy as np

  def forward_pass(w1, w2, x):
      z1 = np.dot(w1, x)
      y1 = np.maximum(z1, 0)
      z2 = np.dot(w2, y1)
      y2 = np.maximum(z2, 0)
      
      return y1, y2, z1, z2

  def backward_pass(y2, y, z1, z2, w2, x):
      # MSE as loss function
      L = 0.5 * (y2 - y)**2
      dldy2 = y2 - y
      # ReLU Derivative
      dy2dz2 = (z2 > 0).astype(float) 
      dldz2 = dldy2 * dy2dz2
      
      dz2dy1 = w2
      dldy1 = np.dot(dz2dy1.T, dldz2)
      # ReLU Derivative
      dy1dz1 = (z1 > 0).astype(float)
      dldz1 = dldy1 * dy1dz1
      
      dldw1 = np.outer(dldz1, x)
      dldw2 = np.outer(dldz2, y1)
      
      return dldw1, dldw2





* ii. Implement vanilla dropout for the forward and backward pass in NumPy.
