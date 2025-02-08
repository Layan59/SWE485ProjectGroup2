import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data: Create a DataFrame (replace this with your actual dataset)
data = {
    'Disease': ['Fungal Infection', 'Allergy', 'GERD', 'Diabetes', 'Jaundice'],
    'Symptoms': [
        ['itching', 'skin rash', 'nodal skin eruptions'],
        ['continuous sneezing', 'shivering', 'chills'],
        ['stomach pain', 'acidity', 'vomiting'],
        ['fatigue', 'weight loss', 'increased appetite'],
        ['itching', 'vomiting', 'fatigue']
    ]
}

# Explode the list of symptoms into separate rows
df = pd.DataFrame(data)
df = df.explode('Symptoms')

# Frequency Distribution of Diseases
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Disease', order=df['Disease'].value_counts().index)
plt.title('Frequency Distribution of Diseases')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap of Symptoms and Diseases
symptom_matrix = pd.crosstab(df['Disease'], df['Symptoms'])
plt.figure(figsize=(12, 8))
sns.heatmap(symptom_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Symptoms and Diseases')
plt.xlabel('Symptoms')
plt.ylabel('Diseases')
plt.tight_layout()
plt.show()