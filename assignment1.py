import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# QUESTION 1

# load in CCSA data
ccsa_data = pd.read_csv('CCSA.csv')
# ccsa_data.head()

# a) Generate a line plot of the monthly CCSA averages versus the reported months. Both
# axes must be correctly labeled, and the tick marks must be appropriately formatted to receive full
# credit.

# first, create a new dataset with the month and average CCSA value for that month and year
ccsa_data['Week'] = pd.to_datetime(ccsa_data['observation_date'], format='%Y-%m-%d').dt.isocalendar().week
ccsa_data['Year'] = pd.to_datetime(ccsa_data['observation_date'], format='%Y-%m-%d').dt.year
ccsa_data['Month'] = pd.to_datetime(ccsa_data['observation_date'], format='%Y-%m-%d').dt.month
ccsa_data['Week_of_Month'] = ccsa_data['Week_of_Month'] = ((ccsa_data['observation_date'].dt.day - 1) // 7) + 1

ccsa_monthly_avg = ccsa_data.groupby(['Year', 'Month'])['CCSA'].mean().reset_index()
ccsa_monthly_avg['Time'] = pd.to_datetime(ccsa_monthly_avg[['Year', 'Month']].assign(DAY=1))
# plot the monthly averages
plt.figure(figsize=(10, 6))
plt.plot(ccsa_monthly_avg['Time'], ccsa_monthly_avg['CCSA'], marker='o')
plt.title('Monthly Average CCSA Values')
plt.xlabel('Time (Months)')
plt.ylabel('Average CCSA')
plt.grid()
plt.show()


# b) Determine the Pearson correlations among the weekly counts. To this end, calculate the
# pairwise Pearson correlation between the counts of the rth and sth weeks of each month, where r
# and s run from 1 to 5 inclusively. Use all available non-missing counts for each pair to calculate the
# correlations. Present the correlations in a 5 × 5 matrix with proper row and column labels.

# pivot the data to have weeks as columns
ccsa_pivot = ccsa_data.pivot_table(index=['Year', 'Month'], columns='Week_of_Month', values='CCSA')

# calculate the Pearson correlation matrix
correlation_matrix = ccsa_pivot.corr(method='pearson')

# display correlation matrix as a heatmap with correlation values overlayed
# note: I used copilot to help generate this heatmap using the comment above as a prompt
plt.figure(figsize=(8, 6))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson Correlation Coefficient')
plt.xticks(range(5), range(1, 6))
plt.yticks(range(5), range(1, 6))
for (i, j), val in np.ndenumerate(correlation_matrix.values):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
plt.title('Pearson Correlation Matrix of Weekly CCSA Counts', pad=20)
plt.show()


# c) Assign observations from 2000 to 2015, inclusive, to the Training partition. The Testing
# partition will contain the remaining ten years of observations. Because not all months have a fifth
# week, further split the Training partition by month into two groups. The first group includes months
# with fifth-week counts. The second group comprises months with only four weeks of counts.
# Generate principal components for the two groups in the Training partition. How many principal
# components do you choose for each group? Also, what percentage of the total variance do the
# principal components explain in each group? Please show your work to justify your choices.

ccsa_train = ccsa_data[(ccsa_data['Year'] >= 2000) & (ccsa_data['Year'] <= 2015)]
ccsa_test = ccsa_data[(ccsa_data['Year'] > 2015)]

# create groups
ccsa_train_pivot = ccsa_train.pivot_table(index=['Year', 'Month'], columns='Week_of_Month', values='CCSA')
train_w_fifth = ccsa_train_pivot[ccsa_train_pivot[5].notna()].dropna()
train_wo_fifth = ccsa_train_pivot[ccsa_train_pivot[5].isna()].drop(columns=[5]).dropna()


# for months with fifth week
pca_5 = PCA()
pca_5.fit(train_w_fifth)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pca_5.explained_variance_ratio_) + 1),
    pca_5.explained_variance_ratio_,
    marker='o'
)
plt.title('Scree Plot: Months with Fifth Week')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, 6))
plt.grid()
plt.show()


# for months without fifth week
pca_4 = PCA()
pca_4.fit(train_wo_fifth)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pca_4.explained_variance_ratio_) + 1),
    pca_4.explained_variance_ratio_,
    marker='o'
)
plt.title('Scree Plot: Months without Fifth Week')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, 5))
plt.grid()
plt.show()

# print explained variance ratios
print("Explained variance ratios for months with fifth week:")
print(pca_5.explained_variance_ratio_)
print("Explained variance ratios for months without fifth week:")
print(pca_4.explained_variance_ratio_)

# based on the elbow plots, I would choose one principle component because it explains almost all of the variance
# (~ 99%)
pca_5.n_components = 1
pca_5.fit(train_w_fifth)

pca_4.n_components = 1
pca_4.fit(train_wo_fifth)

# d) Apply the principal components to the Testing partition. Of course, apply the principal
# components generated from months with only four weeks of counts to the months in the Testing
# partition that have four weeks. Likewise, apply the principal components generated from months
# with five weeks of counts to the months in the Testing partition that also have five weeks. Finally,
# concatenate the principal components’ values in the Testing partition by year and month. What are
# the variances of the principal components applied to the Testing partition?

# create groups in testing partition
ccsa_test_pivot = ccsa_test.pivot_table(index=['Year', 'Month'], columns='Week_of_Month', values='CCSA')
test_w_fifth = ccsa_test_pivot.dropna()
test_wo_fifth = ccsa_test_pivot[ccsa_test_pivot[5].isna()].drop(columns=[5]).dropna()

# transform testing data using trained pca models
test_w_fifth_pca = pca_5.transform(test_w_fifth)
test_wo_fifth_pca = pca_4.transform(test_wo_fifth)

# concatenate the principal components’ values in the Testing partition by year and month
test_w_fifth_pca_df = pd.DataFrame(test_w_fifth_pca, index=test_w_fifth.index, columns=[f'PC1'])
test_wo_fifth_pca_df = pd.DataFrame(test_wo_fifth_pca, index=test_wo_fifth.index, columns=[f'PC1'])
test_pca_concat = pd.concat([test_w_fifth_pca_df, test_wo_fifth_pca_df]).sort_index()

# print the variances of the principal components applied to the Testing partition
print("Variances of the principal components in the Testing partition:")
print(test_pca_concat.var())


# QUESTION 2

from matplotlib.ticker import AutoLocator, MultipleLocator
from mlxtend.frequent_patterns import (apriori, association_rules)
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
import pandas as pd

# For illustrative purposes, we use only observations where the retail theft occurred in 2025 (i.e., Year is 2025) 
# and at the following locations: DEPARTMENT STORE, SMALL RETAIL STORE, GROCERY FOOD STORE, DRUG STORE, CONVENIENCE STORE, GAS STATION, TAVERN/LIQUOR STORE, COMMERCIAL / BUSINESS OFFICE, RESTAURANT, APPLIANCE STORE, WAREHOUSE, and TAVERN / LIQUOR STORE. 
# This selection yielded 13,282 observations for analysis, with an overall arrest rate of approximately 30.65%. 
# We extracted the month and hour of the incident from the Date field. Because the Chicago Police patrol unit operates in three shifts, 
# called Police Watch, we will map each hour to a Police Watch as follows. Police Watch is ‘First’ if the hour is 5 to 11, 
# ‘First -> Second’ if the hour is 12, ‘Second’ if the hour is 13 to 19, ‘Second -> Third’ if the hour is 20, ‘Third’ if the hour is 21 to 23 and 0 to 3,
# and ‘Third -> First’ if the hour is 4

# load in the data
retail_data = pd.read_csv('Retail_Theft_20260109 (1).csv')

# Create the dataset as described in the assignment

# Ensure Date is datetime
retail_data['Date'] = pd.to_datetime(retail_data['Date'])

df_rt = retail_data[
    (retail_data['Year'] == 2025)
].copy()

# Allowed locations
allowed_locations = [
    'DEPARTMENT STORE',
    'SMALL RETAIL STORE',
    'GROCERY FOOD STORE',
    'DRUG STORE',
    'CONVENIENCE STORE',
    'GAS STATION',
    'TAVERN/LIQUOR STORE',
    'COMMERCIAL / BUSINESS OFFICE',
    'RESTAURANT',
    'APPLIANCE STORE',
    'WAREHOUSE',
    'TAVERN / LIQUOR STORE'
]

df_rt = df_rt[
    df_rt['Location Description'].isin(allowed_locations)
]

df_rt['Month'] = df_rt['Date'].dt.month
df_rt['Hour'] = df_rt['Date'].dt.hour

def police_watch(hour):
    if 5 <= hour <= 11:
        return 'First'
    elif hour == 12:
        return 'First->Second'
    elif 13 <= hour <= 19:
        return 'Second'
    elif hour == 20:
        return 'Second->Third'
    elif hour in [21, 22, 23, 0, 1, 2, 3]:
        return 'Third'
    elif hour == 4:
        return 'Third->First'
    else:
        return 'Unknown'

df_rt['Police Watch'] = df_rt['Hour'].apply(police_watch)

df_final = df_rt[['Arrest', 'Location Description', 'Month', 'Police Watch', 'Beat']].copy()
df_final = df_final.dropna()

df_train = df_final[df_final['Beat'].isin([111, 112, 113])].copy()
df_test = df_final[df_final['Beat'] == 1834].copy()
df_train = df_train.drop(columns = ['Beat'])
df_test = df_test.drop(columns = ['Arrest', 'Beat'])

# a) (10 points) Discover association rules with a minimum Support of five incidents and a minimum Lift
# of one. We only want association rules whose Consequents contain a particular 1-item set. That 1-
# item set is either {Arrest is True} or {Arrest is False}. After removing redundant rules, how many
# association rules remain with {Arrest is True} in the consequent? How many association rules remain
# with {Arrest is False} in the consequent?

# preprocess data for association rule mining
df_train['Arrest'] = df_train['Arrest'].astype(str)
transactions = df_train.apply(lambda row: [f"{col}={row[col]}" for col in df_train.columns], axis=1).tolist()  

# encode transactions into indicator format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
retail_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# find frequent itemsets
prop_support = 5/retail_encoded.shape[0]
frequent_itemsets = apriori(retail_encoded, min_support=prop_support, use_colnames=True)

# generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# filter rules for Arrest=True and Arrest=False in the consequent
rules_arrest_true = rules[rules['consequents'].apply(lambda x: 'Arrest=True' in x)]
rules_arrest_false = rules[rules['consequents'].apply(lambda x: 'Arrest=False' in x)]

# Simplify rules by removing redundant rules with lower confidence
# Class code
def simplify_rule (rule_df):
    rule_sorted = rule_df.sort_values(by = 'confidence', ascending = False)
    all_index = rule_sorted.index
    n_rule = rule_sorted.shape[0]
    index_to_drop = []
    for i in range(n_rule):
        s1 = rule_sorted.iloc[i]['antecedents']
        for j in range(i+1,n_rule):
            s2 = rule_sorted.iloc[j]['antecedents']
            if (s1.issuperset(s2)):
                index_to_drop.append(all_index[j])
    simplified_rule_df = rule_sorted.drop(index_to_drop)
    return simplified_rule_df

# remove redundant rules
rules_arrest_true_nr = simplify_rule(rules_arrest_true)
rules_arrest_false_nr = simplify_rule(rules_arrest_false)

# find number of non-redundant rules
print(f"Number of non-redundant rules with Arrest=True in the consequent: {len(rules_arrest_true_nr)}")
print(f"Number of non-redundant rules with Arrest=False in the consequent: {len(rules_arrest_false_nr)}")


# b) (10 points) Plot the Support values against the Confidence values of the association rules of interest
# in Part (b). Use the checkmark ('$\u2713$') if an arrest is made and the 'x' marker if no arrest is
# made. Also, color-code the markers according to the lift values in the rules.

plt.figure(figsize=(10, 6))
scatter_true = plt.scatter(
    rules_arrest_true_nr['confidence'],
    rules_arrest_true_nr['support'],
    c=rules_arrest_true_nr['lift'],
    cmap='viridis',
    marker='$\u2713$',
    label='Arrest=True'
)
scatter_false = plt.scatter(
    rules_arrest_false_nr['confidence'],
    rules_arrest_false_nr['support'],
    c=rules_arrest_false_nr['lift'],
    cmap='viridis',
    marker='x',
    label='Arrest=False'
)
plt.colorbar(scatter_true, label='Lift Value')
plt.title('Support vs Confidence of Association Rules')
plt.ylabel('Support')
plt.xlabel('Confidence')
plt.legend()
plt.grid()
plt.show()


# c)(10 points) Apply the association rules to the testing partition and produce the Confusion Matrix. 
# What is the accuracy? Also, what percentage of observations do the association rules fail to deliver 
# any definitive predictions for?

# Prepare Test Transactions WITHOUT the Arrest feature
# drop Arrest from the features so the rules predict it
test_features = df_test.drop(columns=['Arrest'])
test_transactions = test_features.apply(
    lambda row: [f"{col}={row[col]}" for col in test_features.columns],
    axis=1
).tolist()

predictions = []
actual_labels = df_test['Arrest'].astype(int).tolist()

# Combine all non-redundant rules for prediction
all_rules = pd.concat([rules_arrest_true_nr, rules_arrest_false_nr])

# Prediction
for transaction in test_transactions:
    matched_rules = []
    
    # Check if rule antecedents are a subset of the transaction features
    for _, rule in all_rules.iterrows():
        if rule['antecedents'].issubset(set(transaction)):
            matched_rules.append(rule)

    if matched_rules:
        # Resolve multiple matches by choosing the rule with the highest confidence
        best_rule = max(matched_rules, key=lambda x: x['confidence'])
        if 'Arrest=True' in best_rule['consequents']:
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        # Track cases where no rules apply
        predictions.append(None)

# Filter for valid predictions to build the matrix
valid_results = [(p, a) for p, a in zip(predictions, actual_labels) if p is not None]
y_pred, y_true = zip(*valid_results) if valid_results else ([], [])

# Generate the Confusion Matrix
from sklearn.metrics import confusion_matrix as sk_cm

cm_array = sk_cm(y_true, y_pred, labels=[1, 0])
cm_df = pd.DataFrame(
    cm_array, 
    index=['Actual Arrest', 'Actual No Arrest'], 
    columns=['Predicted Arrest', 'Predicted No Arrest']
)

print("Corrected Confusion Matrix:")
print(cm_df)

# Calculate Metrics
accuracy = sum(1 for p, a in valid_results if p == a) / len(valid_results) if valid_results else 0
no_prediction_rate = predictions.count(None) / len(predictions)

print(f"\nAccuracy (excluding 'None'): {accuracy:.2%}")
print(f"No Prediction Rate: {no_prediction_rate:.2%}")


# d) (10 points) Based on the test partition results, what are the top three most invoked rules for
# predicting an arrest? What are the top three most invoked rules for predicting no arrest? For each
# rule, list the rule number, the number of times it was invoked, the rule criterion, the confidence
# value, and the lift value.

rules_arrest_true_nr = rules_arrest_true_nr.reset_index(drop=True)
rules_arrest_false_nr = rules_arrest_false_nr.reset_index(drop=True)

rules_arrest_true_nr['rule_id'] = ['T' + str(i+1) for i in range(len(rules_arrest_true_nr))]
rules_arrest_false_nr['rule_id'] = ['F' + str(i+1) for i in range(len(rules_arrest_false_nr))]

invocation_counts = defaultdict(int)

for transaction in test_transactions:
    for _, rule in rules_arrest_true_nr.iterrows():
        if rule['antecedents'].issubset(set(transaction)):
            invocation_counts[rule['rule_id']] += 1

    for _, rule in rules_arrest_false_nr.iterrows():
        if rule['antecedents'].issubset(set(transaction)):
            invocation_counts[rule['rule_id']] += 1

rules_arrest_true_nr['invocations'] = rules_arrest_true_nr['rule_id'].map(invocation_counts).fillna(0).astype(int)
rules_arrest_false_nr['invocations'] = rules_arrest_false_nr['rule_id'].map(invocation_counts).fillna(0).astype(int)

top3_arrest = rules_arrest_true_nr.sort_values(
    by='invocations',
    ascending=False
).head(3)

top3_no_arrest = rules_arrest_false_nr.sort_values(
    by='invocations',
    ascending=False
).head(3)

cols_to_show = [
    'rule_id',
    'invocations',
    'antecedents',
    'confidence',
    'lift'
]

print("Top 3 Arrest=True Rules")
print(top3_arrest[cols_to_show])

print("Top 3 Arrest=False Rules")
print(top3_no_arrest[cols_to_show])

# e) (10 points) Suppose Susan owns a SMALL RETAIL STORE in downtown Chicago, and she will hire a
# private security guard to protect her store from retail theft. Based on your rules, when should Susan
# request the security guard be on duty, particularly when the police are less available?

susan_specific_rules = rules_arrest_false_nr[
    rules_arrest_false_nr['antecedents'].apply(lambda x: 'Location Description=SMALL RETAIL STORE' in x)
].copy()
susan_specific_rules = susan_specific_rules.sort_values(by='confidence', ascending=False)

print("Rules for SMALL RETAIL STORE leading to No Arrest:")
print(susan_specific_rules[['antecedents', 'consequents', 'confidence', 'lift']])