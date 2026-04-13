import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data.csv")

# Multiple inputs
X = df[["age", "experience"]]

# Output
y = df["salary"]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[29, 4]])

print("Predicted Salary:", prediction[0])
## Output: Predicted Salary: 56063.15789473684
## The above code demonstrates a multiple linear regression model that predicts salary based on both age and experience. The dataset is loaded from a CSV file, and the model is trained using the `LinearRegression` class from the `sklearn` library. Finally, a prediction is made for a person who is 29 years old with 4 years of experience, resulting in a predicted salary of $56063.15789473684
   
## Matplotlib Visualization (Model Accuracy)
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Prepare data
plt.scatter(df["age"], df["experience"])
plt.xlabel("Age")
plt.ylabel("Experience")
plt.title("Age vs Experience")
plt.savefig("day2.png")
plt.show()



