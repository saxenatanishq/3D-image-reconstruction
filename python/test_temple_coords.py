import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz

image1 = io.imread("data/im1.png")
image2 = io.imread("data/im2.png")

height1, width1, _= image1.shape
height2, width2,_= image2.shape
M1 = max(height1, width1)
M2 = max(height2, width2)

data = np.load("data/some_corresp.npz")
pts1 = data['pts1']
pts2 = data['pts2']

# 2. Compute fundamental matrix using the eight-point algorithm

F = sub.eight_point(pts1, pts2, M1)
print("Fundamental Matrix:\n", F)

# 3. Load points in image 1 from data/temple_coords.npz

temple_data = np.load("data/temple_coords.npz")
pts1_image = temple_data['pts1']

# 4. Run epipolar_correspondences to get points in image 2

pts2_image = sub.epipolar_correspondences(image1, image2, F, pts1_image)

# hlp.epipolarMatchGUI(image1, image2, F)

# 5. Compute the camera projection matrix P1

intrinsics = np.load("data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']

E = sub.essential_matrix(F,K1,K2) # essential matrix
print("Essential Matrix:\n", E)


# Since it is given that for now we can assume the rotational matrix to be I and translational matrix to be O, so,
X = np.hstack((np.eye(3), np.zeros((3,1))))
#     [1 0 0 0]
# X = [0 1 0 0]       X is the extrinsic matrix
#     [0 0 1 0]
P1 = K1 @ X


# 6. Use camera2 to get 4 possible camera projection matrices P2

Extrinsic_options = hlp.camera2(E)
# Choose the correct P2 by checking valid 3D points


# 7. Run triangulate using the projection matrices
# 8. Figure out the correct P2


best_extrinsic = None
max_valid_points = 0

for ind in range(4):
    E = Extrinsic_options[:,:,ind]
    P2_candidate = K2 @ E  #Compute final projection matrix for second camera using this option of E
