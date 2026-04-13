import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data.csv")

# Multiple inputs
X = df[["age"]]

# Output
y = df["salary"]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[29]])

print("Predicted Salary:", prediction[0])

## Output: Predicted Salary: 57272.727272727265

## The above code demonstrates a simple linear regression model that predicts salary based on age. The dataset is loaded from a CSV file, and the model is trained using the `LinearRegression` class from the `sklearn` library. Finally, a prediction is made for a person who is 29 years old, resulting in a predicted salary of approximately $57,273.

## Matplotlib Visualization (Model Accuracy)
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("data.csv")
# Prepare data
X = df[["age"]]
y = df["salary"]
# Train model
model = LinearRegression()
model.fit(X, y)
plt.scatter(df["age"], df["salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()



