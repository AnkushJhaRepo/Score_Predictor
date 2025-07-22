import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
num_students = 100

study_hours = np.random.normal(5, 2, num_students)
attendance = np.random.normal(80, 10, num_students)
past_scores = np.random.normal(65, 15, num_students)

# Final score formula with some noise
final_scores = (study_hours * 5) + (attendance * 0.3) + (past_scores * 0.4) + np.random.normal(0, 5, num_students)

# Create DataFrame
data = pd.DataFrame({
    'Study Hours': study_hours,
    'Attendance': attendance,
    'Past Scores': past_scores,
    'Final Score': final_scores
})

print(data.head())


X = data[['Study Hours', 'Attendance', 'Past Scores']].values
y = data['Final Score'].values.reshape(-1, 1)

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

n_features = X_norm.shape[1]
w = np.random.randn(n_features, 1)
b = 0.0

m = X_norm.shape[0]
alpha = 0.01
sum_w = np.zeros_like(w)
sum_b = 0.0


#gradient descent
# for i in range(m):
#     sum_w += ((np.dot(w.T, X_norm[i]) + b) - y[i]) * X_norm[i].reshape(-1, 1)
#     sum_b += ((np.dot(w.T, X_norm[i]) + b) - y[i])
# sum_w = sum_w/m
# sum_b = sum_b/m
# w = w - alpha*sum_w
# b = b-alpha*sum_b

#cost function
# cost = 0.0
# for i in range(m):
#     cost += ((np.dot(w.T, X_norm[i]) + b) - y[i]) ** 2
# cost = cost/(2*m)


costs = []
n_iterations = 1000
for iter in range(n_iterations):
    sum_w = np.zeros_like(w)
    sum_b = 0.0
    #gradient descent

    for i in range(m):
        sum_w += ((np.dot(w.T, X_norm[i]) + b) - y[i]) * X_norm[i].reshape(-1, 1)
        sum_b += ((np.dot(w.T, X_norm[i]) + b) - y[i])
    sum_w = sum_w/m
    sum_b = sum_b/m
    w = w - alpha*sum_w
    b = b-alpha*sum_b
    
    if iter % 10 == 0:
        cost = 0.0
        for i in range(m):
            cost += ((np.dot(w.T, X_norm[i]) + b) - y[i]) ** 2
        cost = cost/(2*m)
        #print(f"Iteration {iter}: Cost = {cost.item():.2f}")
        costs.append(cost.item())




plt.plot(range(0, n_iterations, 10), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Over Iterations")
plt.grid(True)
plt.show()



# Predict on training data
y_pred = np.dot(X_norm, w) + b

# Print first 5 predictions vs actual
for i in range(5):
    print(f"Predicted: {y_pred[i].item():.2f} | Actual: {y[i]}")
        

custom_input = np.array([[7, 75, 78.23]])  # shape: (1, n)
custom_input_norm = (custom_input - X_mean) / X_std
custom_pred = np.dot(custom_input_norm, w) + b
print(f"Predicted score: {custom_pred.item():.2f}")
