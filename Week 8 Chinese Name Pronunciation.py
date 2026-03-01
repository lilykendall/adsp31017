# -*- coding: utf-8 -*-
"""
Name: Week 8 Chinese Name Pronunciation.py
Course: ADSP 31017 Machine Learning I
Author: Ming-Long Lam, Ph.D.
Organization: University of Chicago
Last Modified: February 23, 2026
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

import string

from sklearn import naive_bayes

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

def CountAlphabet (astring):
   out_dict = {}
   for alpha in astring:
      if (alpha in out_dict.keys()):
         out_dict[alpha] += 1
      else:
         out_dict[alpha] = 1
   return (out_dict)
 
phonetic_name = pandas.read_excel('Chinese_FirstName_Phonetic.xlsx')

# Analysis specifications
q_lowercase = False
q_remove_column0 = False
q_empirical_threshold = True

# Optionally change the phonetic firstname to lowercase
if (q_lowercase):
   phonetic_name['CountAlphabet'] = phonetic_name['FirstName'].str.lower().apply(CountAlphabet)
   X_train_full = pandas.DataFrame(columns = [u for u in string.ascii_lowercase], dtype = float)
else:
   phonetic_name['CountAlphabet'] = phonetic_name['FirstName'].apply(CountAlphabet)
   X_train_full = pandas.DataFrame(columns = [u for u in string.ascii_lowercase + string.ascii_uppercase], dtype = float)

for alpha_count in phonetic_name['CountAlphabet']:
   X_train_full = pandas.concat([X_train_full, pandas.DataFrame(alpha_count, index = [0])], axis = 0, ignore_index = True) 

X_train_full = X_train_full.fillna(0)

# Optionally remove columns with sums of zero
if (q_remove_column0):
   alpha_sum = X_train_full.sum(axis = 0)
   X_train = X_train_full[alpha_sum[alpha_sum > 0].index]
else:
   X_train = X_train_full.copy()

# Optionally calculate the threshold
if (q_empirical_threshold):
   empirical_proportion = phonetic_name['Gender'].value_counts(normalize = True).sort_index()
   threshold_female_prop = empirical_proportion.loc['Female']
else:
   threshold_female_prop = 0.5

y_train = phonetic_name['Gender'].astype('category')
y_count = y_train.value_counts().sort_index()

my_alpha = 0.5 / y_count.shape[0]

classifier = naive_bayes.MultinomialNB(alpha = 0.01).fit(X_train, y_train)
print('Alpha Value = ', classifier.alpha)

print('Class Count:\n', classifier.class_count_)
print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Feature Count (before adding alpha):\n', classifier.feature_count_)
print('Log Feature Probability:\n', classifier.feature_log_prob_)

y_train_predProb = pandas.DataFrame(classifier.predict_proba(X_train), columns = classifier.classes_)

y_train_predClass = numpy.where(y_train_predProb['Female'] > threshold_female_prop, 'Female', 'Male')

confusion_matrix = pandas.crosstab(y_train, y_train_predClass)
overall_accuracy = 0.0
total_count = 0.0
category_accuracy = {}
for a_cat in classifier.classes_:
   diagonal_count = confusion_matrix.loc[a_cat, a_cat]
   category_sum = confusion_matrix.loc[a_cat].sum()
   category_accuracy[a_cat] = diagonal_count / category_sum

   overall_accuracy += diagonal_count
   total_count += category_sum

confusion_matrix['Category Accuracy'] = category_accuracy
overall_accuracy = overall_accuracy / total_count

print(overall_accuracy)
print(confusion_matrix)