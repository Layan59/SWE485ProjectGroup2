# SWE485Project

My Role in the Project


🔹 Phase 1: Problem Understanding and Data Exploration

[BODOR]
 • Collected and examined a relevant dataset aligned with our project’s goal.
 • Provided samples of the raw dataset within the Jupyter Notebook for transparency.
 • Created visualizations such as histograms and tables to show:
 • Distribution of variables
 • Missing values
 • Statistical summaries (mean, variance, etc.)
 • Helped with initial preprocessing, including identifying and handling missing data.
 • Contributed to writing the dataset summary and description section.

[WAAD]
 • Handled the initial preprocessing of the dataset using Pandas and NumPy.
 • Applied data cleaning techniques such as:
 • Removing columns with only NaN values using dropna(axis=1, how='all')
 • Dropping irrelevant features (e.g., ‘Disease’ column before clustering)
 • Checking for duplicate rows and ensuring data integrity
 • Performed basic data formatting and ensured columns were properly structured.
 • Verified dataset readiness for modeling by inspecting data types and null distributions.

[HUTOON]
I worked on describing the goal of the dataset and its source. This included explaining the purpose of the data and where it was collected from.

[Abrar]
• Handled the preprocessing techniques for the selected dataset.
• Applied data cleaning steps

[LAYAN]
• Created the GitHub repository for the project and organized its folder structure to ensure clear collaboration.
• Added the main dataset folder and uploaded the cleaned version used by all team members.
• Wrote the introductory section in the Jupyter Notebook that includes:
- The project motivation
- Students’ names
- A general overview of the project and its goal

🔹 Phase 2: Supervised Learning

[BODOR]
 • Created a Jupyter Notebook in the “Supervised Learning” folder.
 • Implemented multiple supervised learning algorithms (e.g., SVM, Decision Trees).
 • Wrote code with detailed comments to improve clarity and reusability.
 • Analyzed and interpreted the results of each algorithm to identify the best-performing model.

[WAAD]
 • Finalized the data cleaning and transformation before model training.
 • Encoded categorical features if needed and standardized numerical features using StandardScaler.
 • Ensured the dataset was free of inconsistent entries, empty cells, or formatting issues.
 • Helped prepare a clean dataset that was directly used to train the models in this phase.

[HUTOON]
 I was responsible for testing supervised learning algorithms and comparing their results to help determine which models worked best.

 [Abrar]
 Implemented one of the supervised learning algorithms used in the project.

 [LAYAN]
• Implemented the SVM (Support Vector Machine) algorithm using the training dataset and fine-tuned hyperparameters to improve model accuracy.
• Visualized performance using the confusion matrix and evaluated results using metrics like accuracy and classification report.
• Participated in modifying the SVM model to reduce overfitting by adjusting C, kernel, gamma, and max_iter.
• Collaborated with the team to compare SVM with Random Forest and contributed to the decision-making on which algorithm performs better.

🔹 Phase 3: Unsupervised Learning

[BODOR]
 • Evaluated clustering performance using metrics such as:
 • Silhouette Coefficient
 • Total Within-Cluster Sum of Squares (WCSS)
 • BCubed Precision and Recall
 • Created visualizations (e.g., cluster plots) to demonstrate group separations.
 • Discussed how clustering could enhance recommendations and its limitations.

[WAAD]
 • Created the Jupyter Notebook for this phase and implemented DBSCAN, K-Means, and Agglomerative Clustering (HAC).
 • Used StandardScaler for scaling features and applied PCA for visualizing clustering results in 2D.
 • Produced multiple visualizations (e.g., scatter plots, dendrogram, elbow method).
 • Explained the outcomes and compared the clustering algorithms based on performance metrics.

[HUTOON]
I handled the visualizations for clustering algorithms. I created clear visual representations for K-Means, DBSCAN, and HAC using PCA to show how the data was grouped.

[Abrar]

Designed and implemented the unsupervised learning algorithm (clustering) for the project.
• Preprocessed the dataset for clustering (e.g., removing class labels).

[LAYAN]
• Worked on enhancing the DBSCAN algorithm by adjusting the eps and min_samples parameters to improve clustering performance.
• Ran several experiments to reduce the number of clusters while keeping a reasonable Silhouette Score.
• Tested different configurations and helped the team decide on the most logical DBSCAN setup for interpretation.
• Supported in evaluating and comparing results from K-Means, HAC, and DBSCAN, and summarized findings to support model selection.

🔹 Phase 4: Integrating Generative AI

[BODOR]

Handles the user interface where the user enters symptoms.
Prepares the symptoms as input data for the prediction model.

[WAAD]

Documents the entire Phase 4 work.
Compares the outputs of both response templates and justifies which one is more effective.

[HUTOON]

Created the Generative_AI folder.
Added the gpt_response_template.md file with two response formats (Formal and Friendly).
Wrote the gpt_response.py script to connect the predicted disease to GPT using the OpenAI API.
Prepared the code to use the API key from GitHub Secrets.
Added a placeholder disease name until the model is connected.

[Abrar]

Uses the trained machine learning model to predict the disease from the user input.
Passes the predicted disease name to the GPT response script.

[LAYAN]

Adds the API key to GitHub Secrets and completes the integration with GPT.