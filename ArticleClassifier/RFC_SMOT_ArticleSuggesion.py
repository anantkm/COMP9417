# COMP9417 Machine Learning
# Project Multi-class text classification
# Group: Renamed_Group_42
#  ____________________________________
#  |Group Member's Name     |   zID   |
#  |-----------------------------------
#  |Anant Krishna Mahale    |z5277610 |
#  |Hongyi Gao              |z5183218 |    
#  |Nan Wu                  |z5238997 |   
#  |Raghavendran Kasinathan |z5284284 |
#  |Yeqin Huang             |z5175742 |

#Importing the required Libraries 
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE     #open source library from sklearn for handling class-imbalance. [Link:https://imbalanced-learn.readthedocs.io/en/stable/install.html]
#------------------------------------------------------------------------------------------------

#Helper functions. 
def roundof (x):
    return round(x,3)
#------------------------------------------------------------------------------------------------
#setting the start time 
start_time = time.time() 

#Importing and preprocessing the Data.
training_file = 'training.csv'
test_file = 'test.csv'

print('Selected File for Training the Model :',training_file)
print('Selected File for Training the Model :',test_file)

raw_data_train = pd.read_csv(training_file)  #data for training the model. 
raw_data_test = pd.read_csv(test_file)       #Unseen Test data. 


#setting the article number to the index.
raw_data_train.set_index('article_number', inplace=True)   
raw_data_test.set_index('article_number', inplace=True)

#converting the pandas dataframe to Numpy array.
X_train_temp = raw_data_train['article_words'].to_numpy()   
y_train = raw_data_train['topic'].to_numpy()


X_test_temp = raw_data_test['article_words'].to_numpy()
y_test = raw_data_test['topic'].to_numpy()

#converting each words to columns. 
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train_temp)
X_test = count_vect.transform(X_test_temp)
#------------------------------------------------------------------------------------------------
#Since there is a Class-Imabalance, Applying SMOT to training data. 
print("Applying SMOT on Training data...")
sm = SMOTE(random_state=42)
X_rs, y_rs = sm.fit_resample(X_train, y_train)

#------------------------------------------------------------------------------------------------
print("Training the Model with Random Forest Classifier...")
#RF CLASSIFIER MODEL TRAINING
rfc_model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=42,
                       verbose=0, warm_start=True)
rfc_model.fit(X_rs, y_rs)
predicted_y_rfc = rfc_model.predict(X_test)
pobabilities_test = rfc_model.predict_proba(X_test)
print("Priting Accuracy and Classification Report...")
print("\nAccuracy Score: {:.2f}".format(accuracy_score(y_test, predicted_y_rfc)))
print("\nClassfication Report:")
print(classification_report(y_test, predicted_y_rfc))

#------------------------------------------------------------------------------------------------

print("Processing the output...")
#Converting the output and probabilities to Dataframe to provide the required output. 
df_pred = pd.DataFrame(predicted_y_rfc, index=raw_data_test.index, columns=['predicted_class'])
pobabilities_test_df = pd.DataFrame(pobabilities_test, index=df_pred.index)
df_pred = df_pred.merge(pobabilities_test_df, left_index=True, right_index=True, how='left')

#Renaming the columns. 
df_pred.rename(columns={
    0: "ARTS CULTURE ENTERTAINMENT",
    1:"BIOGRAPHIES PERSONALITIES PEOPLE",
    2: "DEFENCE",
    3: "DOMESTIC MARKETS",
    4: "FOREX MARKETS",
    5: "HEALTH",
    6: "IRRELEVANT",
    7: "MONEY MARKETS",
    8: "SCIENCE AND TECHNOLOGY",
    9: "SHARE LISTINGS",
    10:"SPORTS"},inplace = True)

#applying round function to each cell in the probability-output dataframe. 
round_columns=['ARTS CULTURE ENTERTAINMENT',
       'BIOGRAPHIES PERSONALITIES PEOPLE', 'DEFENCE', 'DOMESTIC MARKETS',
       'FOREX MARKETS', 'HEALTH', 'IRRELEVANT', 'MONEY MARKETS',
       'SCIENCE AND TECHNOLOGY', 'SHARE LISTINGS', 'SPORTS']
for col in round_columns:
    df_pred[col] = df_pred[col].apply(roundof)
output_columns = df_pred

#filtering the output data based on the set threshold. 
ace_articles = output_columns[output_columns['predicted_class']== 'ARTS CULTURE ENTERTAINMENT']
ace_articles = ace_articles[ace_articles['ARTS CULTURE ENTERTAINMENT']>0.7452]

bpe_articles = output_columns[output_columns['predicted_class']== 'BIOGRAPHIES PERSONALITIES PEOPLE']
bpe_articles = bpe_articles[bpe_articles['BIOGRAPHIES PERSONALITIES PEOPLE']>0.75]

defence_articles = output_columns[output_columns['predicted_class']== 'DEFENCE']
defence_articles = defence_articles[defence_articles['DEFENCE']>0.75]

dm_articles = output_columns[output_columns['predicted_class']== 'DOMESTIC MARKETS']
dm_articles = dm_articles[dm_articles['DOMESTIC MARKETS']>0.75]

forex_articles = output_columns[output_columns['predicted_class']== 'FOREX MARKETS']
forex_articles = forex_articles[forex_articles['FOREX MARKETS']>0.7275]

health_articles = output_columns[output_columns['predicted_class']== 'HEALTH']
health_articles = health_articles[health_articles['HEALTH']>0.75]

mmar_articles = output_columns[output_columns['predicted_class']== 'MONEY MARKETS']
mmar_articles = mmar_articles[mmar_articles['MONEY MARKETS']>0.6975]

sat_articles = output_columns[output_columns['predicted_class']== 'SCIENCE AND TECHNOLOGY']
sat_articles = sat_articles[sat_articles['SCIENCE AND TECHNOLOGY']>0.75]

sl_articles = output_columns[output_columns['predicted_class']== 'SHARE LISTINGS']
sl_articles = sl_articles[sl_articles['SHARE LISTINGS']>0.75]

sports_articles = output_columns[output_columns['predicted_class']== 'SPORTS']
sports_articles = sports_articles[sports_articles['SPORTS']>0.74]


print('\nPlease find the suggested articles below :\n')
#Final Output to the User. 
output_topics=['ARTS CULTURE ENTERTAINMENT',
       'BIOGRAPHIES PERSONALITIES PEOPLE', 'DEFENCE', 'DOMESTIC MARKETS',
       'FOREX MARKETS', 'HEALTH', 'MONEY MARKETS',
       'SCIENCE AND TECHNOLOGY', 'SHARE LISTINGS', 'SPORTS']
output_articles =[ace_articles,
                  bpe_articles,
                  defence_articles,
                  dm_articles,
                  forex_articles,
                  health_articles,
                  mmar_articles,
                  sat_articles,
                  sl_articles,
                  sports_articles]

for i in range(0,len(output_articles)):
    print(output_topics[i], ":" ,output_articles[i].index.tolist()[:10])

print("Overall duration of the program : {:.2f} seconds".format(time.time() - start_time))