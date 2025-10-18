# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 00:56:31 2025

@author: Matthew
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import math


data = load_iris()

x = data.data
y = data.target
Y = np.eye(3)[y]


X_train, X_test, Y_train, Y_test = train_test_split(
    x,
    Y,
    test_size=0.3,
    stratify=y,
    random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

W1_shape = (4,3)
b1_shape = (1,3)
W2_shape = (3,3)
b2_shape = (1,3)

np.random.seed(42)
W1 = 0.707 * np.random.randn(*W1_shape) ## sqrt(2/ input_to_layer) -> sqrt(2/4) = 0.707
W2 = 0.816 * np.random.randn(*W2_shape) 
b1 = np.zeros(b1_shape)
b2 = np.zeros(b2_shape)

epochs = 200
losses = []

for epoch in range(epochs):
    Z1 = X_train_std @ W1 + b1
    A1 = np.maximum(0, Z1)
    
    Z2 = A1 @ W2 + b2
    z_shifted = Z2 - np.max(Z2, axis = 1, keepdims=True)
    exp_vals = np.exp(z_shifted)
    P = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    loss = -np.mean(np.sum(Y_train * np.log(P + 1e-12), axis=1))
    
    dZ2 = (1 / len(P)) * (P - Y_train)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T 
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X_train_std.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    learning_rate = 0.2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 5 == 0:
        losses.append(loss)
        y_pred = np.argmax(P, axis=1)
        y_true = np.argmax(Y_train, axis=1)
        acc = np.mean(y_pred == y_true)
        print(F"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {acc:.4f}")
    



