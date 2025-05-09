
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    'hour': [0, 4, 8, 12, 16, 20],
    'day_of_week': [1, 1, 2, 3, 4, 5],
    'cpu_usage': [0.2, 0.35, 0.45, 0.5, 0.3, 0.25]
}
df = pd.DataFrame(data)

# Model training
X = df[['hour', 'day_of_week']]
y = df['cpu_usage']
model = LinearRegression()
model.fit(X, y)

# Prediction
test_input = [[10, 2]]
predicted = model.predict(test_input)[0]

# Visualization
y_pred = model.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', color='red', marker='x')
plt.title('Actual vs Predicted CPU Usage')
plt.xlabel('Sample Index')
plt.ylabel('CPU Usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")

# Save results to HTML
with open("results.html", "w") as f:
    f.write(f'''
    <html>
    <head><title>Minimal Proof of Concept</title></head>
    <body>
        <h2>Predicted CPU usage for hour=10, day_of_week=2: {predicted:.2f}</h2>
        <img src="actual_vs_predicted.png" alt="CPU Prediction Plot" width="600">
    </body>
    </html>
    ''')
