import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Raw Data: Create a DataFrame with inconsistencies
data = {
    'Disease': ['Fungal Infection', 'Allergy', 'GERD', 'Diabetes', None, 'Allergy', 'Jaundice', 'Diabetes'],
    'Symptoms': [
        ['itching', 'skin rash', 'nodal skin eruptions'],
        ['continuous sneezing', 'shivering', 'chills'],
        ['stomach pain', 'acidity', None],  # Missing symptom
        ['fatigue', 'weight loss', 'increased appetite'],
        ['itching', 'vomiting'],  # Duplicate disease
        ['continuous sneezing', 'shivering', 'chills'],  # Duplicate symptoms
        ['itching', 'vomiting', 'fatigue'],
        ['fatigue', 'weight loss', 'increased appetite']  # Duplicate entries
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Explode the list of symptoms into separate rows, handling None values
df = df.explode('Symptoms')

# Drop rows where Disease is NaN
df = df.dropna(subset=['Disease'])

# Frequency Distribution of Diseases (with uncleaned data)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Disease', order=df['Disease'].value_counts().index)
plt.title('Frequency Distribution of Diseases (Uncleaned Data)')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap of Symptoms and Diseases (after exploding)
symptom_matrix = pd.crosstab(df['Disease'], df['Symptoms'])
plt.figure(figsize=(12, 8))
sns.heatmap(symptom_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Symptoms and Diseases (Uncleaned Data)')
plt.xlabel('Symptoms')
plt.ylabel('Diseases')
plt.tight_layout()
plt.show()