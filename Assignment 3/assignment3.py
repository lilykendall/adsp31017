# %% [markdown]
# ### Assignment 3
# Lily Kendall

# %% [markdown]
# You use the Student Academic Success.csv, which was used in the lectures. The categorical predictors are Attendance, Course, Debtor, Displaced, Educational special needs, Gender, International, Marital status, Scholarship holder, and Tuition fees up to date. The continuous predictors are Age at enrollment and Curricular units 1st sem (approved). The label variable is Target, with three categories: Dropout, Enrolled, and Graduate. The reference category is Graduate.

# %%
import pandas as pd
from CHAID import Tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
# import Dr. Lam's MultinominalLogisticRegression function from LogisticRegression.py file
from LogisticRegression import MultinominalLogisticRegression

# %%
df = pd.read_csv("Student Academic Success.csv", delimiter=";")
df.head()

# %%
df.columns

# %%
# Columns specified in the assignment instructions
df_keep = df[["Daytime/evening attendance", "Course", "Debtor", "Displaced", "Educational special needs",
              "Gender", "International", "Scholarship holder", "Marital status", "Tuition fees up to date", 
              "Age at enrollment", "Curricular units 1st sem (approved)", "Target"]]
df_keep.head()

# %%
# no missing values, so no need to impute
df_keep.isna().sum()

# %% [markdown]
# ***Question 1***
# 
# You'll build a simple CHAID tree with a depth of 1 for each categorical predictor that has three or more categories. Afterward, you'll create a table showing the counts of terminal nodes for each of these trees. In this table, each row will represent a node, and you'll include the original predictor categories for each node to make everything clear. Each column will correspond to a label category, helping you easily see the distribution.

# %%
# find the number of categories in each categorical variable
categorical_cols = ["Daytime/evening attendance", "Course", "Debtor", "Displaced", "Educational special needs", "Gender", "International", "Marital status", "Scholarship holder", "Tuition fees up to date"]
for col in categorical_cols:
    print(f"{col}: {df_keep[col].nunique()} categories")

# %%
# Distribution of course
df_keep.groupby('Course').describe()

# %%
# Distribution of marital status
df_keep.groupby('Marital status').describe()

# %%
# course and marital status have > 3 categories, so we will create CHAID with those
# starting with course
course_tree = Tree.from_pandas_df(df_keep, {'Course': 'nominal'}, 'Target', dep_variable_type='categorical', max_depth=1)

# %%
course_tree.print_tree()
course_tree.classification_rules()

# %%
# marital status tree
marital_tree = Tree.from_pandas_df(df_keep, {'Marital status': 'nominal'}, 'Target', dep_variable_type='categorical', max_depth=1)

marital_tree.print_tree()
marital_tree.classification_rules()

# %% [markdown]
# ***Question 2***
# 
# You'll train a CHAID tree with a depth of 1 for each continuous predictor, considering each unique value as its own category. To make sure each bin has enough observations, you'll set a maximum limit for each predictor. For instance, for Age at enrollment, values over 55 will be capped at 55 (meaning whichever is smaller between 55 and the actual Age at enrollment). Similarly, for Curricular units 1st sem (approved), the cap will be at 15. 
# 
# Afterward, you'll create a table showing the counts of terminal nodes for each of these trees. In this table, each row will represent a node, and you'll include the original predictor values for each node to make everything clear. Each column corresponds to a label category, making it easy to see the distribution.
# 

# %%
# age at enrollment tree
# first, cap age at 55
df_keep['Age at enrollment'] = df_keep['Age at enrollment'].clip(upper=55)
# create bins for each unique age value/make it categorical
df_keep['Age bins'] = df_keep['Age at enrollment'].astype('category')

age_tree = Tree.from_pandas_df(df_keep, {'Age bins': 'ordinal'}, 'Target', dep_variable_type='categorical', max_depth=1)
age_tree.print_tree()
age_tree.classification_rules()

# %%
# curricular units tree
# first, cap curricular units at 15
df_keep['Curricular units 1st sem (approved)'] = df_keep['Curricular units 1st sem (approved)'].clip(upper=15)
# create bins for each unique value/make it categorical
df_keep['Curricular units bins'] = df_keep['Curricular units 1st sem (approved)'].astype('category')

curricular_tree = Tree.from_pandas_df(df_keep, {'Curricular units bins': 'ordinal'}, 'Target', dep_variable_type='categorical', max_depth=1)
curricular_tree.print_tree()
curricular_tree.classification_rules()

# %% [markdown]
# ***Question 3***
# 
# You will train a multinomial logistic regression using the Forward Selection method with an entry threshold of 5%. The predictors are the original categorical predictors with exactly two categories, as well as the CHAID tree leaves. You should create a step summary table that includes step numbers. At each step, include the predictor’s name, the log-likelihood value, the model degrees of freedom, the Chi-square test statistic, the test degrees of freedom, and the significance value of the test.

# %%
mnlogit_df = df_keep[["Daytime/evening attendance", "Debtor", "Displaced", "Educational special needs", "Gender", "International", "Scholarship holder", "Tuition fees up to date", "Target"]].copy()

# make course tree into category
groups={0: [1, 7], 1: [2, 11, 14, 4, 13], 2: [3, 15], 3: [5, 6], 4: [8, 17], 5: [9, 16], 6: [10, 12]}
mnlogit_df['Course group'] = df_keep['Course'].map(lambda x: next((k for k, v in groups.items() if x in v), None))
mnlogit_df['Course group'] = mnlogit_df['Course group'].astype('category')

# make marital status tree into category
groups={0: [1, 3], 1: [2, 5, 4, 6]}
mnlogit_df['Marital status group'] = df_keep['Marital status'].map(lambda x: next((k for k, v in groups.items() if x in v), None))
mnlogit_df['Marital status group'] = mnlogit_df['Marital status group'].astype('category')

# make age tree into category
groups={0: [17, 18, 19], 1: [20], 2: [21, 22], 3: [23, 24], 4: [25, 26, 27, 28], 5: [29, 30, 31], 6: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]}
mnlogit_df['Age group'] = df_keep['Age bins'].map(lambda x: next((k for k,v in groups.items() if x in v), None))
mnlogit_df['Age group'] = mnlogit_df['Age group'].astype('category')

# make curricular units tree into category
groups={0: [0], 1: [1, 2], 2: [3], 3: [4], 4: [5], 5: [6, 7, 8], 6: [9, 10, 11, 12], 7: [13, 14], 8: [15]}
mnlogit_df['Curricular units group'] = df_keep['Curricular units 1st sem (approved)'].map(lambda x: next((k for k, v in groups.items() if x in v), None))
mnlogit_df['Curricular units group'] = mnlogit_df['Curricular units group'].astype('category')

mnlogit_df.head()

# %%

# Setup lists of predictors
binary_preds = [
    'Daytime/evening attendance', 'Debtor', 'Displaced', 
    'Educational special needs', 'Gender', 'International', 
    'Scholarship holder', 'Tuition fees up to date'
]
tree_groups = ['Course group', 'Marital status group', 'Age group', 'Curricular units group']

candidates = binary_preds + tree_groups
selected_preds = []
step_summary = []

# Calc baseline log-likelihood 
counts = df['Target'].value_counts()
n_total = len(df)
# Log-likelihood of null model
current_llk = np.sum(counts * np.log(counts / n_total))

# Calculate Baseline Degrees of Freedom
# J is number of categories in Target. Intercept-only DF = (J - 1)
n_categories = len(counts)
current_df = n_categories - 1

print(f"Baseline LLK: {current_llk}")
print(f"Baseline DF: {current_df}")

# Forward selection loop
while len(candidates) > 0:
    best_p_value = 1.0
    best_predictor = None
    best_metrics = {}

    for predictor in candidates:
        # Try adding the predictor to current selected set
        trial_preds = selected_preds + [predictor]
        
        # Fit the trial model
        trial_fit, trial_llk, trial_df, *_ = MultinominalLogisticRegression(
            trainData=mnlogit_df, catPred=trial_preds, intPred=[], nominalLabel='Target', 
            modelSpec=trial_preds, qIntercept=True
        )

        # Calculate test statistics
        chi_sq = 2 * (trial_llk - current_llk)
        test_df = trial_df - current_df
        p_value = stats.chi2.sf(chi_sq, test_df)

        if p_value < best_p_value:
            best_p_value = p_value
            best_predictor = predictor
            best_metrics = {
                'LLK': trial_llk,
                'ModelDF': trial_df,
                'ChiSq': chi_sq,
                'TestDF': test_df,
                'Sig': p_value
            }

            # Print step summary for this predictor
            print(f"Trying to add: {predictor}")
            print(f"  LLK: {trial_llk}")
            print(f"  Model DF: {trial_df}")
            print(f"  Chi-square: {chi_sq}")
            print(f"  Test DF: {test_df}")
            print(f"  p-value: {p_value}")

    # Check the 5% Entry Threshold
    if best_p_value < 0.05:
        selected_preds.append(best_predictor)
        candidates.remove(best_predictor)
        current_llk = best_metrics['LLK']
        current_df = best_metrics['ModelDF']
        
        step_summary.append({
            'Step': len(selected_preds),
            'Predictor': best_predictor,
            'Log-Likelihood': current_llk,
            'Model DF': current_df,
            'Chi-square': best_metrics['ChiSq'],
            'Test DF': best_metrics['TestDF'],
            'Significance': best_metrics['Sig']
        })
    else:
        # No more predictors meet the 5% threshold
        break

summary_df = pd.DataFrame(step_summary)

# %%
display(summary_df)

# %% [markdown]
# ***Question 4***
# 
# You will present the confusion matrix and the row accuracy for the Question 3 model. How does this model compare to the one discussed in the Week 6 lecture?

# %%
# fit final model with selected predictors
model_fit, model_llk, model_df, model_param, alias_param, nonalias_param, labelCategories = MultinominalLogisticRegression(
    trainData=mnlogit_df, catPred=selected_preds, intPred=[], nominalLabel='Target', 
    modelSpec=selected_preds, qIntercept=True
)

# Get the predicted probabilities from final model_fit
predicted_probs = model_fit.predict() 

# Find the index of the highest probability for each observation
predicted_indices = predicted_probs.argmax(axis=1)

# Map the indices back to the actual category names
predicted_labels = [labelCategories[i] for i in predicted_indices]

# Get the actual labels
actual_labels = mnlogit_df['Target']

# %%
# Create confusion matrix
conf_matrix_df = pd.crosstab(actual_labels, predicted_labels, 
                             rownames=['Actual'], colnames=['Predicted'])

# Heatmap
cm = confusion_matrix(actual_labels, predicted_labels, labels=labelCategories)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelCategories)

disp.plot(cmap=plt.cm.Blues)
plt.title('Final Model Confusion Matrix')
plt.show()

# Calculate Overall Accuracy
# correct predictions / total predictions
overall_accuracy = np.diag(conf_matrix_df).sum() / conf_matrix_df.values.sum()

# Calculate Row Accuracy
# correct predictions / total row count
row_accuracy = np.diag(conf_matrix_df) / conf_matrix_df.sum(axis=1)

# Add row accuracy
conf_matrix_with_metrics = conf_matrix_df.copy()
conf_matrix_with_metrics['Row Accuracy (%)'] = (row_accuracy * 100).round(2)

print("Confusion Matrix with Row Accuracy:")
print(conf_matrix_with_metrics)
print(f"\nOverall Model Accuracy: {overall_accuracy:.2%}")

# %% [markdown]
# ***Question 5:***
# 
# Building on your model from Question 3, could you find out which predictor value combinations would lead the model to predict the highest probability for each label category?

# %%

# Get predicted probabilities for the entire dataset
all_probs = pd.DataFrame(model_fit.predict(), columns=labelCategories)

# Join probabilities with the predictors used in the final model
final_predictors = selected_preds 
analysis_df = pd.concat([mnlogit_df[final_predictors].reset_index(drop=True), all_probs], axis=1)

# Find the combination with the highest probability for each category
top_profiles_list = []

for category in labelCategories:
    # Find the row with the max probability for the category
    idx_max = analysis_df[category].idxmax()
    
    # Extract the full row
    top_row = analysis_df.loc[idx_max].copy()
    
    # Add a column to identify which category this row represents
    top_row['Target Category'] = category
    top_profiles_list.append(top_row)

# Create summary table
top_profiles_table = pd.DataFrame(top_profiles_list)

cols = ['Target Category'] + [cat for cat in labelCategories] + final_predictors
top_profiles_table = top_profiles_table[cols]

print("Predictor Combinations for Highest Predicted Probabilities:")
display(top_profiles_table)


