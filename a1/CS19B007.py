import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# converting the greyscale image into a 256x256 matrix
image = img.imread("85.jpg")


# -------------------Eigen Value Decomposition-----------------------------------------------------

eigen_values, eigen_vectors = np.linalg.eig(image)

# sorting eigen values and arranging corresponding eigen vectors, by sorting a temporary list of (magnitude of eigen values, real part of eigen values, index) tuples
temp1 = []
for i in range(256) : 
    temp1.append((abs(eigen_values[i]),eigen_values[i].real,i))
temp1.sort(reverse=True)

eigen_values_sorted = np.empty(shape=256, dtype=complex)
eigen_vectors_sorted = np.array(eigen_vectors[:,temp1[0][2]])

for i in range(256) : 
    eigen_values_sorted[i] =  eigen_values[temp1[i][2]]
for i in range(1,256) : 
    eigen_vectors_sorted = np.c_[eigen_vectors_sorted, eigen_vectors[:,temp1[i][2]]]

# compute the sorted eigen value matrix, the corresponding eigen vector matrix and its inverse
P = np.copy(eigen_vectors_sorted)
P_inverse = np.linalg.inv(P)
D = np.diag(eigen_values_sorted)


# --------------------Singular Value Decomposition-----------------------------------------------------

# creating a copy of the image matrix into an array with data type as int, so as to prevent overflow when the matrix is multiplied with its transpose
image1 = np.array(image,dtype=int)

# U is the eigen vector matrix of A.(At)
# singular values are the square root of the eigen values of A.(At)
# Vt is obtained from the equation : A = U.S.Vt, by multiplying both sides with U_inverse and S_inverse
U = np.linalg.eig(image1 @ image1.transpose())[1]
singluar_values = np.sqrt(np.linalg.eig(image1 @ image1.transpose())[0])
Vt = np.linalg.inv(np.diag(singluar_values)) @ np.linalg.inv(U) @ image1

# sorting singular values and arranging corresponding right and left singular vectors,
# by sorting a temporary list of (singular values, right singular vector, left singluar vector transpose) tuples
temp2 = []
for i in range(256) : 
    temp2.append((singluar_values[i], U[:,i], Vt[i,:]))
temp2.sort(reverse=True)

singluar_values_sorted = np.empty(shape=(256))
singluar_values_sorted[0] = temp2[0][0]
U_sorted = np.array(temp2[0][1])
Vt_sorted = np.array(temp2[0][2])

for i in range(1,256) : 
    singluar_values_sorted[i] = temp2[i][0]
    U_sorted = np.c_[U_sorted, temp2[i][1]]
    Vt_sorted = np.vstack([Vt_sorted, temp2[i][2]])


# -------------------------------Plots----------------------------------------------------------------

# reconsturcted images and corresponding error images plot

fig, ax = plt.subplots(5, 4, figsize=(12, 20)) 
curr_fig=0
for k in [5, 20, 50, 180, 255]:  
  # incrementing the value of k so as to include the complex conjugate of any complex eigen value
  # checks if the magnitude of kth eigen value is equal to the magnitude of the (k+1)th eigen value and sum of imaginary parts are zero, given they are not purely real
    if((k!=255) and (temp1[k+1][0] == temp1[k][0]) and (eigen_values_sorted[temp1[k+1][2]].imag + eigen_values_sorted[temp1[k][2]].imag == 0) and (eigen_values_sorted[temp1[k][2]].imag != 0)) :
        k += 1

    approx_evd = P[:,:k] @ D[:k,:k] @ P_inverse[:k,:]
    approx_svd = U_sorted[:,:k] @ np.diag(singluar_values_sorted)[:k,:k] @ Vt_sorted[:k,:]
    
    # evd reconstructed images
    ax[curr_fig][0].imshow(approx_evd.real,cmap="gray")
    ax[curr_fig][0].set_title("evd : k = "+str(k))
    ax[curr_fig,0].axis("off")

    # evd error images
    ax[curr_fig][1].imshow((image - approx_evd.real),cmap="gray")
    ax[curr_fig][1].set_title("evd error : k = "+str(k))
    ax[curr_fig,1].axis("off")

    # svd reconstructed images
    ax[curr_fig][2].imshow(approx_svd,cmap="gray")
    ax[curr_fig][2].set_title("svd : k = "+str(k))
    ax[curr_fig,2].axis("off")

    # svd error images
    ax[curr_fig][3].imshow((image - approx_svd),cmap="gray")
    ax[curr_fig][3].set_title("svd error : k = "+str(k))
    ax[curr_fig,3].axis("off")
    
    curr_fig +=1


# plotting the norm of error matrices vs k

k_set = np.arange(1,256,1)
evd_set = []
svd_set = []

# creating lists of norms of error matrices for various values of k
for k in k_set : 
    approx_evd = P[:,:k] @ D[:k,:k] @ P_inverse[:k,:]
    approx_svd = U_sorted[:,:k] @ np.diag(singluar_values_sorted)[:k,:k] @ Vt_sorted[:k,:] 
    evd_set.append(np.linalg.norm(image-approx_evd.real,"fro"))
    svd_set.append(np.linalg.norm(image-approx_svd,"fro"))

# plot of the two norms vs k 
fig = plt.figure(num=2, figsize=(16, 12))
ax = fig.add_subplot(1,1,1)
quality_threshold = np.full((255,1),1000)
plt.plot(k_set,evd_set, label="evd norm")
plt.plot(k_set,svd_set, label="svd norm")
plt.plot(k_set,quality_threshold)

# plotting the intersection at norm value = 1000
first_line = LineString(np.column_stack((k_set, evd_set)))
second_line = LineString(np.column_stack((k_set, svd_set)))
third_line = LineString(np.column_stack((k_set, quality_threshold)))
intersection1 = third_line.intersection(first_line)
intersection2 = third_line.intersection(second_line)

plt.plot(*intersection1.xy, 'ro')
plt.plot(*intersection2.xy, 'ro')

plt.legend()
plt.show()