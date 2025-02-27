import pandas as pd
import matplotlib.pyplot as plt

# Load the raw and cleaned datasets
raw_data = pd.read_csv(r'C:\Users\Asus\SWE485ProjectGroup2-main\Dataset\datasetDiseaseSymptomPrediction.csv')  # Original dataset
cleaned_data = pd.read_csv(r'C:\Users\Asus\SWE485ProjectGroup2-main\Code\cleaned_dataset.csv')  # Cleaned dataset

# Summary statistics before cleaning
raw_summary = {
    'Dataset': 'Raw Data',
    'Rows': raw_data.shape[0],
    'Columns': raw_data.shape[1],
    'Missing Values': raw_data.isnull().sum().sum()
}

# Summary statistics after cleaning
cleaned_summary = {
    'Dataset': 'Cleaned Data',
    'Rows': cleaned_data.shape[0],
    'Columns': cleaned_data.shape[1],
    'Missing Values': cleaned_data.isnull().sum().sum()
}

# Create a DataFrame for the summaries
summary_df = pd.DataFrame([raw_summary, cleaned_summary])

# Plotting the summary as a table
plt.figure(figsize=(8, 4))
plt.axis('tight')
plt.axis('off')
plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
plt.title('Summary of Raw vs Cleaned Data')
plt.show()

# Plotting the comparison as a bar chart
summary_df.set_index('Dataset').plot(kind='bar', figsize=(10, 6), alpha=0.7)
plt.title('Comparison of Raw and Cleaned Data')
plt.ylabel('Count')
plt.xlabel('Dataset')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()