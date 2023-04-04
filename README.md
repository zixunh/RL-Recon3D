<!-- %%Zixun Huang 3038564193 Apr 3rd%%
 -->

## 1. How to calculate Fundamental Matrix
### Epipolar Constraint
Based on epipolar geometry, we have $x_{2}^TFx_{1}=0$, where $F$ is the fundamental matrix, and 

$$
x_{2}^T=
\begin{bmatrix}
u' & v'& 1 \\
\end{bmatrix}
$$

$$
x_{1}=
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
$$

$$
F=
\begin{bmatrix}
f_{11}&f_{12}&f_{13}\\
f_{21}&f_{22}&f_{23}\\
f_{31}&f_{32}&f_{33}
\end{bmatrix}
$$


### Eight (pair of) points algorithm
We can first rewrite it into:
$$u'uf_{11}+u'vf_{12}+u'f_{13}+v'uf_{21}+v'vf_{22}+v'f_{23}+uf_{31}+vf_{32}+f_{33}=0$$

$$
\tilde{x} \cdot \vec{f} =
\begin{bmatrix}
u'u&u'v&u'&v'u&v'v&v'&u&v&1 \\
\end{bmatrix}
\begin{bmatrix}
f_{11}\\
f_{12}\\
f_{13}\\
f_{21}\\
f_{22}\\
f_{23}\\
f_{31}\\
f_{32}\\
f_{33}
\end{bmatrix} = 0
$$

That is, $\vec{f}$ represents the fundamental matrix in the form of a 9-dim vector, and it must be (approximately) orthogonal to the vector $\tilde{x}$ which is constructed by a pair of corresponding points. 
To solve this equation, we need to use 8 point algorithom, that is, at least 8 pairs of corresponding points should be involved. Each pair can derive an equation based on epipolar geometry. By stacking these equations, then we will have a linear system $A\vec{f}=0$, where each row entry of $A$ is constructed by a pair of matches. And $f$ is a non-zero vector and belongs to the Null-space of this matrix $A$. i.e. $\vec{f} \in N(A)$ s.t. $\lvert \lvert \vec{f} \rvert \rvert\neq 0$. Since we don't care the scale of fundamental matrix, we just set $$\lvert \lvert \vec{f} \rvert \rvert=1$$.

```python
'''
	here we use x,y,x_,y_ to indicate u,v,u',v'
	% matches(i,0:1) is a point in the first image
	% matches(i,2:3) is the corresponding point in the second image 
'''
x,y,x_,y_ = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
'''concat''' 
A = np.stack((x_*x, x_*y,x_,y_*x,y_*y,y_,x,y),axis=1)
A = concat((A, np.ones((A.shape[0],1))), axis=1)
```

### Approximate solutions based on SVD
While $rank(A)$ should be 8, if there is no noise, but in real world, we would see $rank(A)>8$, i.e. there is no exact solution to this linear system.
This equation should be overdetermined if we have enough correspondings. To solve it, we can see if we do SVD on $A \in R^{N\times9}$, the linear system becomes $U\Sigma V^T\vec{f}=0 \to \Sigma V^T\vec{f}=0$, where $V \in R^{9 \times 9}$. An approximate solution can be found by solving $argmin_{\vec{f}}\lvert \lvert \Sigma V^T\vec{f} \rvert \rvert_{2}^2$.

We can approximately set $f$ as the transpose of the row entries of $V^T$, i.e. column vector of $V$, corresponding to the smallest singular value in $\Sigma$. 
1. Because $V$ is an orthogonal matrix, the row entries of $V^T$ are orthogonal with each other, i.e. the dot product between should be zero. By set $\vec{f}$ as one singular vector, we will have 

$$
V^Tf=[0,0,0,0,0,0,0,0,\lvert\lvert\vec{f}\rvert\rvert_{2}^2]^T=
\begin{bmatrix}
0_{8 \times 1}\\ 
1 \\ 
\end{bmatrix}_{9 \times1}
$$

2. Assume singular values in $\Sigma$ is sorted, the optimal minimal value of this expression will be determined by the last (smallest) singular value $\sigma_{9}=\sigma_{min}$. 

$$
\lvert \lvert \Sigma V^T\vec{f} \rvert  \rvert_{2}^2 =\lvert \lvert  
\begin{bmatrix}
diag(\sigma_{i})_{9 \times 9}\\
0_{(N-9)\times9}
\end{bmatrix}_{N \times 9}
\begin{bmatrix}
0_{8 \times 1}\\ 
1 \\ 
\end{bmatrix}_{9 \times1} 
\rvert  \rvert_{2}^2=\sigma_{min}^2$$

3. After solve $f$, we can reshape it back to $F\in R^{3 \times3}$.
4. Think about essential matrix $E=t\wedge R=[t_{\times}]R$, where $[t_{\times}]$ has rank 2. So we need to enforce $F$ has rank 2 as well.
```python
def fundamental_matrix(matches):
    '''normalize'''
    x,y,x_,y_ = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    t1 = normalize_matrix(x,y)
    t2 = normalize_matrix(x_,y_)
    x,y,x_,y_ = normalize(x), normalize(y), normalize(x_), normalize(y_)
    '''concat''' 
    A = np.stack((x_*x, x_*y,x_,y_*x,y_*y,y_,x,y),axis=1)
    A = concat((A, np.ones((A.shape[0],1))),axis=1)
    '''solve SVD'''
    u,s,vh = np.linalg.svd(A)
    f = vh[-1, :]
    F = f.reshape(3,3)
    '''rank = 2'''
    u,s,vh = np.linalg.svd(F)
    s[-1]=0
    F = u@np.diag(s)@vh
    '''denormalize'''
    F = t2.T@F@t1
    return F
```

### Normalization and Denormalization
If we need to do normalization on (non-homogeneous) points before we do eight point algorithm, we can assume nomalized points are $\tilde{x}=\frac{x-\mu}{\sigma}\gamma$, where $\gamma$ is the scale factor.
For homogeneous space, it's equivalent to $\tilde{x}=Tx$. that is:

$$
\begin{bmatrix}
\tilde{u}\\ 
\tilde{v}\\ 
1\\
\end{bmatrix} = 
\begin{bmatrix}
\frac{\gamma_{u}}{\sigma_{u}} & 0 & -\frac{\gamma_{u}\mu_{u}}{\sigma_{u}}\\ 
0 & \frac{\gamma_{v}}{\sigma_{v}} & -\frac{\gamma_{v}\mu_{v}}{\sigma_{v}}\\ 
0 & 0 & 1\\
\end{bmatrix}
\begin{bmatrix}
u\\ 
v\\ 
1\\
\end{bmatrix}
$$

After nomalization, we can continue to solve $\tilde{x_{2}}^T\tilde{F}\tilde{x_{1}}=0$. But remember to denormalize it after we get $\tilde{F}$ from the SVD. Since $\tilde{x_{2}}^T\tilde{F}\tilde{x_{1}}=x_{2}^TT_{2}^T\tilde{F}T_{1}x_{1}=0$, we can get $F=T_{2}^T\tilde{F}T_{1}$.
```python
def normalize(x, r=np.sqrt(2)):
    return (x-np.mean(x))/(np.std(x))*r #, np.mean(x), np.std(x)
def normalize_matrix(x,y,r=np.sqrt(2)):
    T = np.stack(([r/np.std(x), 0, -r*np.mean(x)/np.std(x)],
                [0, r/np.std(y), -r*np.mean(y)/np.std(y)],
                [0, 0, 1]), axis = 1)
    return T
```


### Residual Error



### Deliverables
##### Here is how to approximately solve the fundamental matrix step by step: 
1. normalize the corresponding points
2. concat the epipolar constraints provided by those matches into a linear system
3. use SVD to approximately solve the linear system
4. reshape and enforce the fundamental matrix to rank 2
5. denormalize

##### Is the Residual Error what we are directly optimizing using SVD when solving the homogeneous system? If yes, explain. If no, how does the objective relate to the residual?
- It's not the identical to the thing we are directly optimizing, but they are highly related.
- The geometric representation of 8-point algorithm:
	- What we are optimizing in 8-point algorithm is the coplanarity of 3 vectors w.r.t the relative camera orientation:
		- The first vector is the translation between two camera. $^0t$
		- Second, the ray passing through the point on image1 from the first camera (center of projection). $^1Rx_{1}$ 
		- Then, the ray passing through the corresponding point on image2 from the second camera. $^2x_{2}$
	- In other words, we pick the optimal relative camera orientation so that the 2 rays $^{1,2}$ approximately intersect. 
- Then **what are we directly optimizing**? Is it equivalent to the residual?
	- Think about the epipolar geometry with calibated cameras, what we are optimizing is the following, where $\theta'=\frac{\pi}{2}-\theta$ is the angle between the vector $x_{2}$ and the optimal epipolar plane spanned by $t$ and $Rx_{1}$. 
	- $$
	\lvert x_{2}^T[T_{\times}]Rx_{1} \rvert=
	\lvert\lvert x_{2} \rvert\rvert *\lvert\lvert t\wedge Rx_{1} \rvert\rvert*\lvert cos\theta \rvert=
	\lvert\lvert x_{2} \rvert\rvert *\lvert\lvert t\wedge Rx_{1} \rvert\rvert*\lvert\sin\theta' \rvert=
	\lvert\lvert t\wedge Rx_{1} \rvert\rvert*d_{2\to1}
	$$

- So we can see the term $\lvert \lvert x_{2} \rvert \rvert *\lvert\sin\theta' \rvert=d_{2\to1}$  is actually the distance between the point in image2 and the corresponding epipolar plane.
	- We can further derive the remaining term into $\lvert \lvert t\wedge Rx_{1} \rvert\rvert = \lvert \lvert t\rvert \rvert *\lvert \lvert Rx_{1} \rvert\rvert*\lvert\sin\beta \rvert$. This is the distance between the second camera (center of projection) and the ray $Rx_{1}$. This term might look useless for optimization goal, even though we can see that the whole expression turns to zero when $\beta=0$, i.e., $x_{1}$ overlaps its epipolar, $d_{1\to2}=0$.
	- Now we can prove that the expression we are optimizing is not equivalent to the residual error. When $\beta=0$ and $\theta'\neq 0$, the epipolar constraint $\lvert x_{2}^T[T_{\times}]Rx_{1} \rvert$ equals zero, while the residual error $(d_{1\to2}^2+d_{2\to1}^2)/2$ equals $\frac{1}{2}d_{2\to1}^2$ which may not be zero.
	
- What's **the relation between this objective and the residual**?
	- We've already see the epipolar constraint is $\lvert \lvert t\wedge Rx_{1} \rvert\rvert*d_{2\to1}$, while the residual is $(d_{1\to2}^2+d_{2\to1}^2)/2$. Both of them are minimizing the distance between the points and the corresponding epipolar lines.
	- The main difference is that the former is evaluating the coplanarity of  $t$, $Rx_{1}$, $x_{2}$, while the latter is evaluating the dist between points and their corresponding epipolar planes. 
	- One simple case to distinguish these two is the case which I mentioned above. When $x_{1}$ overlaps its epipolar, i.e. the translation $t$ is co-linear with $Rx_{1}$, and $x_{2}$ is at an angle to its epipolar plane, i.e. $\theta'\neq 0$, we can see:
		- $coplanarity=x_{2}^TFx_{1}=0$, but $residual>0$. when $\theta'\neq 0$ and $\beta=0$.


## 2. Find Relative Camera Orientation (R|t)
### Essential Matrix
Recall Essential Matrix is calibrated version of $F$.
$$E=K_{2}^TFK_{1}$$

