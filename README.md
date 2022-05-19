# amazon-food-review
This project aims to classify amazon food reviews. The code includes the following steps. 

## Data cleaning:
### Stemming: words will be transformed to its stem. 
### Punctuation removal: Punctuation is not needed in this amazon review analysis. 
### Lower case

## Feature Engineering
### Text to vector:
#### Unigram: one word is taken as an input
#### Bigram: two words together are taken as an input. 
Python package：
from sklearn.feature_extraction.text import CountVectorizer 

## Model Building
### Logistic Regression
### Random Forest
Python package：
from sklearn.linear_model import LogisticRegression from sklearn.ensemble import RandomForestClassifier  

## Model Evaluation
### ROC AUC
### precision recall F1

## Feature Importance
select important features from logistic regression and random forest

## Restore model 
use pickle to restore models. 
