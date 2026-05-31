import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = sns.load_dataset("iris")
df = pd.DataFrame(data)

# Check missing values and data types
print(df.isnull().sum().sort_values(ascending=False))
print(df.dtypes)

# Encode categorical columns
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype == "category":
        df[col] = encoder.fit_transform(df[col])

print(df.head())

# Prepare features and target
x = df.drop("species", axis=1)
y = df["species"]

# Split data: 80% training, 20% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(xtrain, ytrain)

# Make predictions
pred = model.predict(xtest)

# Calculate accuracy
accuracy = accuracy_score(pred, ytest)
print(accuracy)