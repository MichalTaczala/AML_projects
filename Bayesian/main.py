import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from LDA import LDA
from QDA import QDA
from NB import NB

a = [0.1, 0.5, 1, 2, 3, 5]
p = 2


def generate_data_schema_1(a, p):
    n = 1000

    y = np.random.binomial(1, p, n)
    X = np.zeros((n, 2))

    # Assign features based on the class
    for i in range(n):
        if y[i] == 0:
            X[i] = np.random.normal(0, 1, 2)
        else:
            X[i] = np.random.normal(a, 1, 2)
    return X, y


def generate_data_schema_2(a):
    n = 1000
    p = 2
    y = np.random.binomial(1, 0.5, n)
    X = np.zeros((n, p))

    # Assign features based on the class
    for i in range(n):
        if y[i] == 0:
            X[i] = np.random.normal(0, 1, p)
        else:
            X[i] = np.random.normal(a, 1, p)
    return X, y


a = [2]
p = [0.5]
X, y = generate_data_schema_1(a, p)

lda = LDA()
lda.fit(X, y)
res = lda.predict(X)
for i in range(len(res)):
    print(f"Predicted: {res[i]}, Actual: {y[i]}")


plt.figure(figsize=(10, 6))
plt.scatter(X[res == 0][:, 0], X[res == 0][:, 1], color="red", label="Class 0")
plt.scatter(X[res == 1][:, 0], X[res == 1][:, 1], color="blue", label="Class 1")
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

# Calculate corresponding y values
# a, b = lda.get_line()
# y_values = a * x_values + b

# Plot the line
# plt.plot(x_values, y_values, color="green", label="Decision Boundary")
plt.title("Generated Dataset - Schema1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
