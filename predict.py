import joblib
import sys
import pandas as pd

model = joblib.load("model.joblib")

if len(sys.argv) != 6:
    print("Usage: python predict.py HoursStudied AttendancePercent Assignments SleepHours PreviousMarks")
    sys.exit(1)

vals = list(map(float, sys.argv[1:]))
df = pd.DataFrame([vals], columns=["HoursStudied", "AttendancePercent", "Assignments", "SleepHours", "PreviousMarks"])
pred = model.predict(df)[0]

print(f"Inputs: {dict(zip(df.columns, vals))}")
print(f"🎯 Predicted Marks: {pred:.2f}")
