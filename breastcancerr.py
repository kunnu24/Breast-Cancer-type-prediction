

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import load_breast_cancer
cancer_dataset=load_breast_cancer()
print(cancer_dataset)
print(cancer_dataset.keys())
type(cancer_dataset) 



print(cancer_dataset["feature_names"]) 




can_df=pd.DataFrame(np.c_[cancer_dataset["data"],cancer_dataset["target"]],
                   columns=np.append(cancer_dataset["feature_names"],["target"]))
can_df 





can_df.to_csv("breast_cancer_dataframe.csv")
can_df.head() 




can_df.info() 



can_df.describe() 

sns.pairplot(can_df,hue="target") 



sns.countplot(can_df["target"]) 



sns.countplot(can_df["mean radius"]) 




plt.figure(figsize=(15,15))
sns.heatmap(can_df)  



plt.figure(figsize=(30,30))
sns.heatmap(can_df.corr(),annot=True,cmap="hot",linewidths=3)  


X=can_df.drop(["target"],axis=1)
y=can_df["target"]  




from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0) 
print(X_train)
print(X_test)
print(y_train) 
print(y_test) 

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_train  


from sklearn.svm import SVC
classifier=SVC() 
classifier.fit(X_train,y_train) 




y_pred=classifier.predict(X_test)
y_pred 




from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm)
print(accuracy_score(y_pred,y_test) )
classification_report(y_pred,y_test)


from  sklearn.linear_model import LogisticRegression 
lg_classifier=LogisticRegression(random_state=0)
lg_classifier.fit(X_train,y_train) 





y_pred=lg_classifier.predict(X_test)
y_pred 




from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# # KNN classifier 



from sklearn.neighbors import KNeighborsClassifier
KN_classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2) 
KN_classifier.fit(X_train,y_train) 





y_pred=KN_classifier.predict(X_test)
y_pred 




from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# # Naive bayes classifier



from sklearn.naive_bayes import GaussianNB 
NB_classifier=GaussianNB()
NB_classifier.fit(X_train,y_train) 





y_pred=NB_classifier.predict(X_test) 
y_pred  




from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# # DEcision Tree classifier 



from sklearn.tree import DecisionTreeClassifier
D_classifier=DecisionTreeClassifier(criterion="entropy",random_state=0) 
D_classifier.fit(X_train,y_train) 



y_pred =D_classifier.predict(X_test)
y_pred 


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 


# # random forest classifier 



from sklearn.ensemble import RandomForestClassifier 
rc_classifier=RandomForestClassifier(n_estimators=20,criterion="entropy",random_state=0)
rc_classifier.fit(X_train,y_train) 




y_pred=rc_classifier.predict(X_test)
y_pred 



from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_pred,y_test) 
print(cm) 
accuracy_score(y_pred,y_test) 





from sklearn.model_selection import cross_val_score 
cross_validation=cross_val_score(estimator=rc_classifier,X=X_train,y=y_train) 
cross_validation
print("cross validation mean accuracy",cross_validation.mean()) 





import pickle 
pickle.dump(rc_classifier,open("breast_cancer.pickle","wb"))
breast_cancer_model=pickle.load(open("breast_cancer.pickle","rb"))
y_pred=breast_cancer_model.predict(X_test)
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))

