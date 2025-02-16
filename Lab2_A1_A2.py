import pandas as pd
import numpy as np

file_path = "C:/Users/ashwi/Desktop/Sem 4/ML/Lab_ML/Purchase data.xlsx"
df = pd.read_excel(file_path)

A = df.iloc[:, :-1]
C = df.iloc[:, -1]

# Convert A to numeric format
A = A.apply(pd.to_numeric, errors='coerce')  #Converting non-numeric to NaN
A.fillna(0, inplace=True)  #Replacing NaN values with 0

# Compute dimensionality of vector space (Number of columns in A)
dimensionality = A.shape[1]
print("Dimensionality of the vector space:", dimensionality)

# Computing number of vectors in the vector space (Number of rows in A)
num_vectors = A.shape[0]
print("Number of vectors in the vector space:", num_vectors)

#Computing the rank of A
rank = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank)

#Computing the pseudo-inverse of A
pinv = np.linalg.pinv(A)

# Computing the cost
X = np.dot(pinv, C)
X = np.round(X).astype(int)
print("Estimated cost of products:\n", X)
