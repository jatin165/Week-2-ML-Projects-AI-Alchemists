import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\devul\Downloads\student_exam_scores.csv")

print(df.head())

df['result'] = df['exam_score'].apply(lambda x: 1 if x >= 40 else 0)

X = df[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))