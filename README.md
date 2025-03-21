# 3D Image Recontruction

# Step-by-step go through

---

## 1. Implement the eight-point algorithm

---

1. First, we collect the points from the image and then normalize the points
The method that I have used for normalization is as shown below:

    <p align="center">
    <img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image.png" width="400px">
    </p>
    
2. The **Transformation Matrix** will look like this -

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/fbd3538a-8162-49fc-aabd-58d8d4f63347.png" width="300px">
</p>

1. For each of the points in the pts arrays, we have this equation

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%201.png" width="600px">
</p>

*How we get the fundamental matrix? I saw the derivations from this video
https://youtu.be/6kpBqfgSPRc?si=Rs_kLs6nxVptgKgO*

1. Upon rearranging the terms of the above equation, we can write it as shown below. But this creates a **110×9** matrix (there are 110 points in the pts1 and pts2 array), meaning we have an **overdetermined system** (more equations than unknowns).

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%202.png" width="500px">
</p>

1. Now, in order to find the solution of this overdetermined system of equation, we need to sue SVD

## Using Singular Value Decomposition (SVD)

---

For any matrix A (of size m×n), **SVD** decomposes it into three matrices:

$$
F = USV^T
$$

Where:

- U is an **orthogonal matrix** (m×m) — its columns are the **left singular vectors**
- S is a **diagonal matrix** (m×n) containing **singular values** (ordered from largest to smallest)
- $V^T$ is an **orthogonal matrix** (n×n) — its rows are the **right singular vectors**

In numpy we can write

```python
U, S, Vt = np.linalg.svd(F)
```

From SVD properties, the last column of V is the solution of $Af = 0$

```python
F = Vt[-1].reshape(3, 3)
```

Now we force the matrix to be rank 2 by setting the smallest singular value in F to zero. We do this because a **valid** fundamental matrix **must be rank-2** (since it represents a mapping between two images in epipolar geometry). However, the computed **F** from SVD may have **full rank (rank-3)** due to noise.
To do that, we apply SVD on F again and then manually make the smallest row of S to zero. (The last row is the smallest b default)

Now, to unnormalize F matrix, we use the T1 and T2 transformation matrices. Note that the transformation matrices are different for both the images.

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%203.png" width="600px">
</p>

**Now, we have found the Fundamental Matrix!!!**

In my case the fundamental matrix was something like this,

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%204.png" width="400px">
</p>

1. Using the `displayEpipolarF` function in `python/helper.py` , 

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/Screenshot_2025-03-18_161929.jpg" width="600px">
</p>

## 2. Finding epipolar correspondences

---

To find the corresponding point in the image 2 for the points given in image 1 (There are 288 points given in image 1), we use the fundamental matrix that we just calculated. Using that we can find the equation of a line.

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%205.png" width="600px">
</p>

Now, in order to find the corresponding point in image 2, we use the SSD algorithm (Sum of Squared Difference).

### SSD (Sum of Squared Difference)

---

First, we define a constant called `range = 20`  (range kind of depends on the angle through which the camera is rotated) and we iterate for every integer value of x in the range x2 = x1 - 20 to x1 + 20 and find the corresponding y2 using the equation of the epipolar line that we already know from the fundamental matrix. Then we define another constant `window_size = 5` and for each point we define a square patch around the point x2, y2 of the length and breadth `2*window_size + 1`. Then, we subtract the intensities of the patch 1 and patch 2 (and square them to make it positive).
`error = np.sum((patch1.astype(np.float32) - patch2.astype(np.float32))**2)`

Then we select the x2, y2 with the minimum value of error. This makes sense because if the points are similar then they will have very less difference in the intensities of all points in the window size.

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%206.png" width="500px">
</p>

*Note: Every point in the image 1 will have a corresponding point in image 2 even if no image matches to the actual image. In case the image is not matching any point then we should increase the range (typically it is 20, that's why I have also kept it 20)*

The testing of this function using `epipolarMatchGUI` in `python/helper.py` looks like -

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%207.png" width="600px">
</p>

## 3. Finding Projection Matrix

---

Ideally the projection matrix is the product of intrinsic and extrinsic matrix. But, in our case, we can assume that the extrinsic matrix that is R = I and T = O.

*To understand the P matrix better, I saw this video 
https://youtu.be/qByYk6JggQU?si=mDD4hDKuJgxb39RV*

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%208.png" width="500px">
</p>

Finding P1 is easy but for finding P2, the `hlp.camera2` function returns us 4 P2 matrices and we have to choose the best one. To do that we need to find the Essential Matrix (point 4) and we need to do triangulation (point 5) for every E matrix in the four we have.

## 4. Writing a function to compute the essential matrix

---

*How to get E from F? I found this video very helpful for that
https://youtu.be/6kpBqfgSPRc?si=j2-OMBICYAAp9A3w*

In the diagram shown below,

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%209.png" width="500px">
</p>

The epipolar constraint is,

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2010.png" width="200px">
</p>

Using the epipolar-constraint and after a lot of manipulation of variables, we have this expression

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2011.png" width="400px">
</p>

We can easily recall that this equation resembles the fundamental matrix eqaution,

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2012.png" width="400px">
</p>

And so we can say that,

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2013.png" width="700px">
</p>

Where K1 and K2 are the properties of the camera and is given

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2014.png" width="400px">
</p>

## 5. Triangulation

---

We have the system of equation as shown below (this is the basic definition of projection matrix, applied for both camera 1 and camera 2 for all the 288 points we have)

<p align="center">
<img src="https://raw.githubusercontent.com/saxenatanishq/Task6/refs/heads/main/photos/image%2015.png" width="200px">
</p>

We can see that using the above expression we can get 4 equations, solving which we can easily know the x, y ,z for all the 288 points.

---

### NOTE

I was not able to complete the remaining task due to time constraint and so, I have submitted the code only till point 4. I have not written the code for triangulation and things after triangulation.
