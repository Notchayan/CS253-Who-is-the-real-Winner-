import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display basic statistics of numerical features
print("Basic Statistics of Numerical Features:")
print(train_data.describe())

# Display the distribution of Education levels
plt.figure(figsize=(10, 6))
sns.countplot(x='Education', hue='Education', data=train_data, palette='Set2', legend=False)
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Distribution of Criminal Cases by Party
plt.figure(figsize=(12, 6))
sns.countplot(x='Criminal Case', hue='Party', data=train_data, palette='tab10')
plt.title('Distribution of Criminal Cases by Party')
plt.xlabel('Number of Criminal Cases')
plt.ylabel('Count')
plt.legend(title='Party', bbox_to_anchor=(1, 1))
plt.show()


# Calculate percentage distribution of parties with candidates having the most criminal records
criminal_records = train_data[train_data['Criminal Case'] > 0]
criminal_records_by_party = criminal_records['Party'].value_counts(normalize=True) * 100

# Plot the percentage distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=criminal_records_by_party.index, y=criminal_records_by_party.values)
plt.title('Percentage Distribution of Parties with Candidates Having the Most Criminal Records')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()

# Find the party with the wealthiest candidate
# Find the party with the wealthiest candidate
wealthiest_party = train_data[train_data['Total Assets'] == train_data['Total Assets'].max()]['Party'].values[0]

# Calculate percentage distribution of parties with wealthy candidates
wealthy_candidates_by_party = train_data['Party'].value_counts(normalize=True) * 100

# Plot the percentage distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=wealthy_candidates_by_party.index, y=wealthy_candidates_by_party.values)
plt.title('Percentage Distribution of Parties with the Most Wealthy Candidates')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.show()


# New feature engineering for training data
train_data['Prefix_Preference'] = train_data['Candidate'].apply(lambda x: 1 if 'Adv.' in x or 'Dr.' in x else 0)
train_data['Constituency_Preference'] = train_data['Constituency ∇'].apply(lambda x: -1 if x.startswith(('ST', 'SC')) else 0)

# New feature engineering for test data
test_data['Prefix_Preference'] = test_data['Candidate'].apply(lambda x: 1 if 'Adv.' in x or 'Dr.' in x else 0)
test_data['Constituency_Preference'] = test_data['Constituency ∇'].apply(lambda x: -1 if x.startswith(('ST', 'SC')) else 0)

# Define features and target
features = ['Constituency ∇', 'Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state', 'Prefix_Preference', 'Constituency_Preference']
target = 'Education'

# Convert categorical variables to numeric using LabelEncoder
le = LabelEncoder()
combined = pd.concat([train_data[features], test_data[features]])

for feature in features:
    le.fit(combined[feature])
    train_data[feature] = le.transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])

# Remove non-numeric columns before calculating correlation
numeric_train_data = train_data.select_dtypes(include=['number'])

# Display correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X_train = train_data[features]
y_train = le.fit_transform(train_data[target])
X_test = test_data[features]

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42) 
rf_classifier.fit(X_train_split, y_train_split)

# Predict on the validation set
val_predictions = rf_classifier.predict(X_val_split)
f1 = f1_score(y_val_split, val_predictions, average='weighted') # Calculate F1 score on validation set
print("F1 Score on Validation Set:", f1)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Convert the numeric predictions back to the original classes
predictions = le.inverse_transform(predictions)

# Write the predictions to a CSV file
submission_df = pd.DataFrame({'ID': test_data['ID'], 'Education': predictions})
submission_df.to_csv('my_submission_rf_improved_2.csv', index=False)
