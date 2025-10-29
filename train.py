import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load dataset
data = pd.read_csv("data/marks.csv")

X = data.drop(columns=["Marks"])
y = data["Marks"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model trained and saved as model.joblib")
