# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 19:27:14 2025

@author: Matthew
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

# Load and convert image to grayscale
A = imread(r"C:\Users\Matthew\Downloads\max&sophie.jpeg")
X = np.mean(A, -1)

plt.figure(figsize=(8, 6))
plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.show()

# Perform SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)

# Loop over different ranks
for j, r in enumerate([5, 20, 100], start=1):
    Xapprox = U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]
    
    plt.figure(j)
    plt.imshow(Xapprox, cmap='gray')
    plt.axis('off')
    plt.title(f'r = {r}')
    plt.show()
    
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
