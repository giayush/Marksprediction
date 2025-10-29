# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# import os

# app = Flask(__name__)

# # Load pre-trained model
# model = joblib.load("model.joblib") if os.path.exists("model.joblib") else None

# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html", result=None, inputs={}, coefs=[], error=None)

# @app.route("/predict", methods=["POST"])
# def predict():
#     if model is None:
#         return render_template("index.html", result=None, inputs={}, coefs=[], error="⚠️ Model not trained yet. Run 'python train.py' first.")

#     try:
#         # Get form inputs
#         hours = float(request.form["hours"])
#         attendance = float(request.form["attendance"])
#         assignments = float(request.form["assignments"])
#         sleep = float(request.form["sleep"])
#         previous = float(request.form["previous"])

#         # Create DataFrame
#         df = pd.DataFrame([[hours, attendance, assignments, sleep, previous]],
#                           columns=["HoursStudied", "AttendancePercent", "Assignments", "SleepHours", "PreviousMarks"])

#         # Predict
#         prediction = model.predict(df)[0]
#         prediction = round(prediction, 2)
        
#         inputs = df.iloc[0].to_dic()
#         coefs = []

#         return render_template("index.html", result=prediction, inputs= inputs, coefs=coefs, error=None)

#     except Exception:
#         return render_template("index.html", result=None, inputs={}, coefs=[], error="Please enter valid numbers.")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Load model
if os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
else:
    model = None

# Feature names (must match training order)
# ...existing code...

# Feature names (must match training order)
FEATURES = ["HoursStudied", "AttendancePercent", "Assignments", "SleepHours", "PreviousMarks"]

@app.route("/", methods=["GET", "POST"])
def home():
    error = None
    result = None
    inputs = {}
    coeffs = []

    if request.method == "POST":
        try:
            # Collect user inputs
            hours = float(request.form["hours"])
            attendance = float(request.form["attendance"])
            assignments = float(request.form["assignments"])
            sleep = float(request.form["sleep"])
            previous = float(request.form["previous"])

            inputs = {
                "HoursStudied": hours,
                "AttendancePercent": attendance,
                "Assignments": assignments,
                "SleepHours": sleep,
                "PreviousMarks": previous,
            }

            # Check model
            if model is None:
                error = "⚠️ Model not loaded. Please ensure 'model.joblib' exists and is trained."
            else:
                # Predict marks
                X = np.array([[hours, attendance, assignments, sleep, previous]])
                prediction = model.predict(X)[0]
                result = round(float(prediction), 2)

                # Extract coefficients safely
                if hasattr(model, "coef_"):
                    coeffs = list(zip(FEATURES, model.coef_))
                elif hasattr(model, "named_steps"):
                    for name, step in model.named_steps.items():
                        if hasattr(step, "coef_"):
                            coeffs = list(zip(FEATURES, step.coef_))
                            break
                else:
                    coeffs = []

        except ValueError:
            error = "❌ Please enter valid numbers between 0-100 for attendance and marks, and positive numbers for other fields."
        except Exception as e:
            error = f"⚠️ System Error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        inputs=inputs,
        coeffs=coeffs,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
