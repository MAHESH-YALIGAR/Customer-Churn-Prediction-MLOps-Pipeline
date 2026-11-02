import joblib

X_train = joblib.load("artifacts/transformed/X_train.pkl")
X_test = joblib.load("artifacts/transformed/X_test.pkl")

y_train = joblib.load("artifacts/transformed/y_train.pkl")
y_test = joblib.load("artifacts/transformed/y_test.pkl")

preprocessor = joblib.load(
    "artifacts/transformed/preprocessor.pkl"
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("Preprocessor:", preprocessor)