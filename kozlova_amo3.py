import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

NUM = 5

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"Печать {NUM} предсказаний:")
for i in range(NUM):
    predicted_val = y_pred[i]
    true_val = y_test[i]
    print(
        f"{i+1}. fact: {iris.target_names[true_val]:<10} | pred: {iris.target_names[predicted_val]:<10} | delta: {true_val-predicted_val}"
    )

# Оценка качества

cross = pd.crosstab(
    y_test,
    y_pred,
    colnames = ['Predicted'],
    rownames = ['Actual']
)
print(f"\nCROSS TABLE:\n{cross}")

print(f"\nCLASSIFICATION REPORT:\n{classification_report(y_test, y_pred)}")

print("\nTEST EXAMPLE:")
test_example = [[1, 1, 1, 1]]
prediction = model.predict(test_example)

predicted_class = iris.target_names[prediction][0]
print(f"prediction for: {test_example}")
print(f"predicted class: {predicted_class}")