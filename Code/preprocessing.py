import pandas as pd

# Load dataset
df = pd.read_csv("Dataset/datasetDiseaseSymptomPrediction.csv") 

symptom_columns = df.columns[1:]  

df["Symptoms"] = df[symptom_columns].values.tolist()

df["Symptoms"] = df["Symptoms"].apply(lambda x: list(set(x) - {"None"}))

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

df["Symptoms"] = df["Symptoms"].apply(lambda x: [str(s) for s in x])

symptom_encoded = mlb.fit_transform(df["Symptoms"])

df_encoded = pd.DataFrame(symptom_encoded, columns=mlb.classes_)
df_cleaned = pd.concat([df[["Disease"]], df_encoded], axis=1)

# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)      

# Display the final processed dataset
print(df_cleaned.head()) 

# Save the cleaned dataset to CSV
df_cleaned.to_csv("cleaned_dataset.csv", index=False)
