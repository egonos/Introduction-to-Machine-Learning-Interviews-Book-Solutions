# Chapter 5

## 5.1.1 Vectors

**1. Dot product**

* i.  What’s the geometric interpretation of the dot product of two vectors?

Dot product is used for understanding how much a vector goes along with the direction with another vector.

* ii. Given a vector  u, find vector  v of unit length such that the dot product of  u and  v is maximum.

max u.v where |v| = 1

u.v = |u||v|cos(theta) => max cos(theta) = 1 when theta = 0

then if we select unit vector v on top of vector u we maximize the dot product of two.

max u.v = |u|

**2. Outer product**

* i. Given two vectors  a=[3,2,1] and  b=[−1,0,1] calculate the outer product  aTb? 

a.T = [3,2,1]T    

 b = [-1,0,1]

<a.T,b> = [[-3,0,3],
           [-2,0,2],
           [-1,0,1]]



* ii. Give an example of how the outer product can be useful in ML.

To compute pairwise distance matrix in Simulated Annealing algorithm (ref: https://towardsdatascience.com/outer-products-a-love-letter-b29a2c2c818e)

**3.** What does it mean for two vectors to be linearly independent?

The vector (say) one can not be represented with the vector two ie. no matter how we play around with vector two multiply divide we can not get the vector one. The reverse is also true. We can not get vector two by using linear coefficients and vector one itself.  

**4.** Given two sets of vectors  A=a1,a2,a3,...,an and  B=b1,b2,b3,...,bm. How do you check that they share the same basis?

If they are in the same basis they could be both represented by the basis vectors. The linear combination of the basis vectors should be sufficient to obtain both A & B.

**5.** Given  n vectors, each of  d dimensions. What is the dimension of their span?

To represent d dimensions we have to have d number of linear independent vectors. For example R2 -> (0,1),(1,0). For the scope of this question, if all n vectors are linearly independent then their span is also d dimensions. If not we have to find set basis of these set. The dimension of their spain is wqual to the number of basis vectors we found. 

**6. Norms and metrics**

* i. What's a norm? What is  L0,L1,L2,Lnorm?

Norm is a metric to compute the distantance between two vectors.

* ii. How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

The metric d(u,v)
 induced by a vector space norm has additional properties that are not true of general metrics. These are:

**Translation Invariance:** d(u+w,v+w)=d(u,v)

**Scaling Property:** For any real number t, d(tu,tv)=|t|d(u,v).

Conversely, if a metric has the above properties, then d(u,0)
 is a norm.

More informally, the metric induced by a norm "plays nicely" with the vector space structure. The usual metric on Rn has the two properties mentioned above. But there are metrics on Rn that are topologically equivalent to the usual metric, but not translation invariant, and so are not induced by a norm.

(ref: https://math.stackexchange.com/questions/38634/difference-between-metric-and-norm-made-concrete-the-case-of-euclid)
