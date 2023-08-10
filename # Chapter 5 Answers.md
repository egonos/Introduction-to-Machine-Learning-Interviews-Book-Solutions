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

a.T = [3,   
       2,
       1]    

 b = [-1,0,1]

<a.T,b> = [[-3,0,3],
           [-2,0,2],
           [-1,0,1]]



* ii. Give an example of how the outer product can be useful in ML.

To compute pairwise distance matrix in Simulated Annealing algorithm (ref: https://towardsdatascience.com/outer-products-a-love-letter-b29a2c2c818e)

**3.** What does it mean for two vectors to be linearly independent?

The vactor (say) one can not be represented with the vector two ie. no matter how we play around with vector two multiply divide we can not get the vector one. The reverse is also true. We can not get vector two by using linear coefficients and vector one itself.  