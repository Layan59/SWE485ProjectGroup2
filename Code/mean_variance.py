import pandas as pd

# Example DataFrame with numerical values
data = {
    'Disease': ['Fungal Infection', 'Allergy', 'GERD', 'Diabetes', 'Jaundice'],
    'Symptom Count': [3, 3, 2, 3, 3],
    'Severity Rating': [4, 3, 5, 4, 2]
}

df = pd.DataFrame(data)

# Calculate Mean
mean_symptom_count = df['Symptom Count'].mean()
mean_severity_rating = df['Severity Rating'].mean()

# Calculate Variance
variance_symptom_count = df['Symptom Count'].var()
variance_severity_rating = df['Severity Rating'].var()

print(f"Mean Symptom Count: {mean_symptom_count}")
print(f"Mean Severity Rating: {mean_severity_rating}")
print(f"Variance in Symptom Count: {variance_symptom_count}")
print(f"Variance in Severity Rating: {variance_severity_rating}")