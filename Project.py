#!/usr/bin/env python
# coding: utf-8

# ## 1. Understand the problem at hand
# 
# The software that is used to profile criminals into categories of their likelyhood to commit future crimes has a racial bias.
# 
# Seems as though the software used did not use an algorithm that applied history of offences but may have just used a race algorithm. (Borden/Prater example.) Rating a defendant’s risk of future crime is often done in conjunction with an evaluation of a defendant’s rehabilitation needs. 
# 
# The score used to determine whether a criminal would commit a future crime proved remarkably unreliable in forecasting violent crime: Only 20 percent of the people predicted to commit violent crimes actually went on to do so.
# We also turned up significant racial disparities, just as Holder feared. In forecasting who would re-offend, the algorithm made mistakes with black and white defendants at roughly the same rate but in very different ways.
# 
# The formula was particularly likely to falsely flag black defendants as future criminals, wrongly labeling them this way at almost twice the rate as white defendants.
# White defendants were mislabeled as low risk more often than black defendants.
# 
# 2. Get Familiar with your data
# 
# Risk scales
# - General recidivism
# - Violent recidivism
# - Recedivism risk screen
# - On Counter-Intuitive Predictions
# - Pretrial misconduct
# 
# Pretrial release risk
# - FTA (Failure To Appear)
# - New Felony arrests
# - Pretrial Failure
# 
# The 137 questions
# 
# Criminogenic need scales
# - Cognitive behavioural
# - Criminal Associates/Peers
# - Criminal involvement
# - Criminal opportunity
# - Criminal personality
# - Criminal thinking self report
# - Current violence
# - Family criminality
# - Financial problems
# 

# In[963]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[964]:


pwd


# In[971]:


pd.set_option('display.max_columns', 999)


# In[975]:


scores = pd.read_csv('data/compas-scores-raw.csv')

parsed = pd.read_csv('data/cox-violent-parsed.csv')

parsed_filt = pd.read_csv('data/cox-violent-parsed_filt.csv')

fairml = pd.read_csv('data/propublica_data_for_fairml.csv')


# In[976]:


scores.RawScore.unique()


# In[977]:


scores.Ethnic_Code_Text.unique()


# Why are there seemingly two African-American columns?

# In[978]:


sns.distplot(scores.RawScore, hist=True,  kde=False, bins=50)


# Why is there such a big gap in the scores?

# In[979]:


scores.RawScore.hist();


# In[980]:


scores.Ethnic_Code_Text.value_counts(normalize=True).plot.barh();


# In[981]:


scores.RawScore.value_counts(normalize=True).plot.barh();


# It seems as though there is no data recorded for the group 'African-Am'

# In[982]:


scores.describe()


# In[983]:


import missingno as msno

msno.matrix(scores);


# In[984]:


scores.head()

scores.columns

#sns.distplot(scores.DecileScore, hist=True,  kde=False, bins=20)

#sns.boxplot()

#scores.AssessmentReason.unique()


# In[986]:


scores = scores.drop(['MiddleName', 'Language', 'FirstName', 'LastName', 'DateOfBirth', 
             'Person_ID', 'AssessmentID', 'Case_ID'], axis=1)


# In[ ]:





# In[ ]:





# In[867]:


#scores.head()


# Dropping the columns 'Middle Name' and 'Language' column as i do not find it very relevant.
# 
# There seems to be data that is missing in the 'Score Text' column

# In[868]:


sns.boxplot(y=scores.RawScore)


# In[869]:


numerical = [
  'RecSupervisionLevel', 'RecSupervisionLevelText',
       'Scale_ID', 'DisplayText', 'RawScore', 'DecileScore', 'ScoreText',
       'AssessmentType', 'IsCompleted', 'IsDeleted'
]


# In[870]:


scores[numerical].hist(bins=15, figsize=(15, 8), layout=(3, 4));


# In[871]:


#scores.groupby("Ethnic_Code_Text")["RawScore"].mean().plot.barh() 


# In[ ]:





# Clear indication that African Americans have a much higher RawScore than any other demographic. 
# 
# Side note: Why are there two columns? Also how is it possible that 'African-Am' column has such high scores but above when we did the count there seemed to be no count for this column?

# In[872]:


scores.groupby("Sex_Code_Text")["RawScore"].mean().plot.barh() 


# In[873]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


Data = pd.read_csv('data/compas-scores-raw.csv', skiprows = False)

plot = sns.barplot(x="Ethnic_Code_Text", y="RawScore", data = Data)

plt.title(f"Raw Score of Different Ethnic Individuals")
plt.xticks(rotation=60)
plt.xlabel("Ethnicities of criminal defendants")
plt.figure(figsize=(20,50))


# In[874]:


plot = sns.catplot(x="DisplayText", y="RawScore", data = Data)

plt.title(f"Decile Score of Different Ethnic Individuals")
plt.xticks(rotation=60)
plt.xlabel("Risks associated with criminal defendants")
plt.figure(figsize=(20,50))


# In[ ]:


plot = sns.catplot(x="Ethnic_Code_Text", y="DecileScore", data = Data)

plt.title(f"Decile Score of Different Ethnic Individuals")
plt.xticks(rotation=60)
plt.xlabel("Ethnicities of Criminal Defendants")
plt.figure(figsize=(20,50))


# In[ ]:





# Just looking at the female vs male scores

# In[875]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

categorical_variables = scores.select_dtypes(
    ['object', 'category']).columns.tolist()

numerical_variables = scores.select_dtypes(include=np.number) 

def means_by_category(col_group, col_calc):
    scores.groupby(col_group)[col_calc].mean().nlargest(10).plot(kind='barh', figsize=(10,10))
    sns.mpl.pyplot.xlabel('Mean values for variable {}'.format(col_calc))

interact(means_by_category, col_group=categorical_variables, col_calc=numerical_variables);


# In[876]:


scores.groupby("ScoreText")["RawScore"].mean().plot.barh() 


# Created a widget to look at the various data. It seems as though 'Risk of failure to Appear' greatly increases the score. Even more so than 'Risk of being violent'. 
# 
# Is that indeed what the graph is telling us?

# The graph above clearly shows that African Americans have a much higher Score than any other demographic. Also why in the data are there two African American columns?: 
# 
# Col_group: Ethnic_score_text
# Col_calc: RawScore

# In[877]:


parsed.head()
parsed.columns


# In[878]:


#parsed.drop(["MiddleName", "Language"], axis=1)


# In[879]:


parsed.groupby("is_violent_recid")["v_decile_score"].mean().plot.barh() 


# How can it be that the violent decile scores can have an average that is so close? Surely the non-violent recidivists should have a much lower average score.

# In[880]:


parsed_filt.head()


# In[ ]:





# In[881]:


msno.matrix(parsed);


# In[882]:


#parsed.drop(["MiddleName", "Language"], axis=1)


# In[883]:


categorical_variables = parsed.select_dtypes(
    ['object', 'category']).columns.tolist()

numerical_variables = parsed.select_dtypes(include=np.number) 

def means_by_category(col_group, col_calc):
    parsed.groupby(col_group)[col_calc].mean().nlargest(10).plot(kind='barh', figsize=(10,10))
    sns.mpl.pyplot.xlabel('Mean values for variable {}'.format(col_calc))

interact(means_by_category, col_group=categorical_variables, col_calc=numerical_variables);


# Something very strange going on with the parsed data. Seems as though the matrix is not readable. Also looking at the parsed data and the filtered parsed data, we need to establish what is the difference between these two sets of data.

# The data 'Parsed' and 'Parsed_filt' need to be compared in order to determine what the difference is in the two sets of data. From this point more focus is being put on the parsed_filt data as it seems cleaner

# In[884]:


#parsed = parsed.dropna()

#Attempted to drop all the missing data but the code did seem to run


# In[885]:


parsed_filt.columns


# In[886]:


msno.matrix(parsed_filt);


# In[940]:


parsed_filt = parsed_filt.drop(['name', 'age', 'dob', 'race', 'age_cat', 'violent_recid', 'id', 'first', 'last', 'sex'], axis=1)


# In[ ]:





# In[888]:


#parsed_filt = parsed_filt.dropna()


# It appears as though there is no data in the column 'violent_recid'
# 
# Also the vr_charge_degree, vr_offense_date and vr_charge_desc data is missing alot of values
# 
# These columns are obviously depended on whether or not the person is a violent recid. However if the value is NA do we not fill that with a value and therefore filling in the data?

# In[889]:


categorical_variables = parsed_filt.select_dtypes(
    ['object', 'category']).columns.tolist()

numerical_variables = parsed_filt.select_dtypes(include=np.number) 

def means_by_category(col_group, col_calc):
    parsed_filt.groupby(col_group)[col_calc].mean().nlargest(10).plot(kind='barh', figsize=(10,10))
    sns.mpl.pyplot.xlabel('Mean values for variable {}'.format(col_calc))

interact(means_by_category, col_group=categorical_variables, col_calc=numerical_variables);

#Violent recid seems to have no data


# Within the data i have noticed that for is violent recidivist African Americans sit somewhere in the middle. However their decile score is the highest and their violent decile score is also the highest

# In[890]:


fairml.columns


# In[1006]:


#fairml.drop(['African_American', 'Asian', 'Hispanic', 'Native_American', 'Other', 'Female'], axis=1)


# In[1007]:


msno.matrix(fairml);


# In[1008]:


fairml.head()


# In[1009]:


fairml.columns


# In[1010]:


fairml.groupby("score_factor")["African_American"].mean().plot.barh() 


# In[1012]:


fairml.groupby("score_factor")["Asian"].mean().plot.barh()


# In[1013]:


fairml.groupby("score_factor")["Hispanic"].mean().plot.barh()


# In[896]:


fairml.groupby("Two_yr_Recidivism")["African_American"].mean().plot.barh()


# In[897]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set(rc={'figure.figsize':(6,6)}) 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[987]:


target_variable = "RawScore"
independent_variables = scores.drop(target_variable, axis=1)

target_variable = scores[target_variable]


# In[988]:


independent_variables.head()


# In[989]:


categorical_columns = independent_variables.select_dtypes([object, "category"]).columns
numerical_columns= independent_variables.select_dtypes(np.number).columns


# In[990]:


categorical_data = independent_variables[categorical_columns]
categorical_data.head()


# In[991]:


numerical_data = independent_variables[numerical_columns]
numerical_data.head()


# In[992]:


categorical_data_codified = pd.get_dummies(
                                    categorical_data, 
                                    drop_first=True,
                                    dtype="int64"
)


# In[993]:


independent_codified = pd.concat([
                            numerical_data,
                            categorical_data_codified # reset index so it matches the numerical
                        ], axis=1
)


# In[994]:


independent_codified.head()


# In[995]:


X_train, X_test, y_train, y_test = train_test_split(independent_codified, 
                                                    target_variable, 
                                                    test_size=0.2,
                                                    random_state=42
                                                   )


# In[996]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[997]:


numerical_X_train = X_train[numerical_columns]
numerical_X_train.head()


# In[998]:


codified_X_train = X_train.drop(numerical_columns, axis=1)
codified_X_train.head()


# In[999]:


numerical_X_train.head()


# In[1000]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(numerical_X_train)


# In[1001]:


(scaler.mean_)


# In[1002]:


scores.columns


# In[1003]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[1004]:


predictions = model.predict(X_test)


# In[1005]:


X = X_test.reset_index().copy()
X["RawScore"] = y_test.tolist()
X["prediction"] = predictions
X.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[952]:


target_variable_1 = "decile_score.1"

independent_variables_1 = parsed_filt.drop(target_variable_1, axis=1)


# In[953]:


numerical_data_1 = parsed_filt.select_dtypes(np.number).fillna(0)
categorical_columns_1 = independent_variables_1.select_dtypes([object, "category"]).columns
numerical_data_1.head()


# In[954]:


categorical_data_1 = independent_variables_1[categorical_columns_1]
categorical_data_1.head()


# In[1025]:


categorical_data_codified_1 = pd.get_dummies(
                                    categorical_data_1, 
                                    drop_first=True,
                                    dtype="int64"
)


# In[1026]:


categorical_data_codified_1.head()


# In[1027]:


independent_codified_1 = pd.concat([
                            numerical_data_1,
                            categorical_data_codified_1 
                        ], axis=1
)


# In[1033]:


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(independent_codified_1, 
                                                    target_variable_1, 
                                                    test_size=0.2,
                                                    random_state=42
                                                   )


# In[1035]:


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
     numerical_data_1[independent_variables_1],
     numerical_data_1[target_variable_1], 
     test_size=0.2,   
     random_state=42
)


# In[1034]:


print('X train', X_train_1.shape)
print('y train', y_train_1.shape)
print('X test', X_test_1.shape)
print('y test', y_test_1.shape)


# In[ ]:


model_1 = LinearRegression()
model_1.fit(X_train_1, y_train_1)


# In[ ]:


predictions_1 = model_1.predict(X_test_1)


# In[ ]:


X_1 = X_test_1.reset_index().copy()
X_1["decile_score.1"] = y_test_1.tolist()
X_1["prediction_1"] = predictions_1
X_1.head()


# In[1014]:


X_2 = fairml.drop("Two_yr_Recidivism", axis=1)  

y_2 = fairml['Two_yr_Recidivism']                 


# In[1015]:


X_2.head()


# In[1016]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import SCORERS
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

import warnings
warnings.simplefilter("ignore")


# In[1017]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2,test_size=0.2)


# In[1018]:


print('X_train: ',X_train_2.shape)
print('X_test: ',X_test_2.shape)
print('y_train: ',y_train_2.shape)
print('y_test ',y_test_2.shape)


# In[1019]:


clf = LogisticRegression()
clf.fit(X_train_2, y_train_2)


# In[1020]:


predictions_2 = clf.predict(X_test_2)

predictions_2[:10]


# In[1021]:


predictions_probabilities = clf.predict_proba(X_test_2)
predictions_probabilities[:10]


# In[1022]:


sns.distplot(predictions_probabilities[:,0], bins= 10, kde = False, label = 'Prob0', axlabel = 'Prob0')
sns.distplot(predictions_probabilities[:,1], bins = 10, kde = False, label = 'Prob1', axlabel = 'Prob1')


# In[1043]:


sns.countplot(results["prediction"]) 


# In[1044]:


sns.countplot(results["target"]) 


# In[1038]:


results = X_test_2.reset_index().copy()
results["target"] = y_test_2.tolist()
results["prediction"] = predictions_2
results = pd.concat([results, probs_df], axis=1)
results[["target", "prediction", 0, 1]].head(20)


# Need to increase the threshold

# In[ ]:


cv_results.mean()

