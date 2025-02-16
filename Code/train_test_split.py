import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
df_cleaned = pd.read_csv("cleaned_dataset.csv")

# Split dataset into train and test
X = df_cleaned.drop(columns=["Disease"])
y = df_cleaned["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save train and test sets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data splitting completed successfully!")
