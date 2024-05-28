## gender = 0 -> male
## car = 0 for no ...
## https://www.kaggle.com/datasets/rohit265/credit-card-eligibility-data-determining-factors?select=dataset.csv


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset.csv')

# wich column has which datatype
print(df.info())

# Create columsn for all the possibilities in Education_type
df_dummies = pd.get_dummies(df['Education_type'], prefix='education')
df_dummies2 = pd.get_dummies(df['Income_type'], prefix="incometype")

incometype = df_dummies2.columns

# Convert boolean values to integers
df_dummies = df_dummies.astype(int)
df_dummies2 = df_dummies2.astype(int)

# add both dummies 
df_dummies = pd.concat([df_dummies2, df_dummies], axis=1)

# Concatenate the original DataFrame with the created columns
df = pd.concat([df, df_dummies], axis=1)

# drop the following columns
df = df.drop(["ID", "Education_type" ,"Work_phone", "Phone", "Email", "Num_children", "Num_family", "Housing_type", "Family_status", "Occupation_type", "Income_type"], axis = 1)


# show wich columsn are still in the data Frame
print(f"\nAlle Parameter die berücksichtigt werden, um zu entscheiden, ob eine Person einen Kredit bekommt, oder nicht:\n\n{df.columns}")

# x and y for training
x,y = df.drop("Target", axis=1), df["Target"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4)

# Use all the GPU -> -1
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(x_train,y_train)

# Accuracy
print(f"\n\nAccuracy: {round(clf.score(x_test,y_test) * 100, 2)}%\n\n")

print("Welcher Parameter beeinflusst die Entscheidung, ob eine Person einen Kredit bekommt und in welchem Ausmaß (in prozent angegeben):\n")

# Which column is the most important 
importances = {}
for feature, importance in zip(clf.feature_names_in_, clf.feature_importances_):
    # round the percent to the second digit 
    importances[feature] = round(importance * 100, 2)

    
value = 0 

# Add all the incomtypes up
for income in importances:
    if income in incometype:
        value = value + importances[income]

value = round(value, 2)

# add the new calculated values to dict
importances['(Income types added together)'] = value

# sort the dict 
sorted_importances =  sorted(importances.items(), key=lambda x: x[1], reverse=True)

for imp in sorted_importances:
    print(imp)