import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.metrics import roc_auc_score


def confu (conf) :
    accuracy = (conf[0][0] + conf[1][1]) / (conf[0][0] + conf[1][0] + conf[0][1] + conf[1][1])
    precision = (conf[1][1] ) / (conf[1][1] + conf[0][1] )
    recall = (conf[1][1]) / (conf[1][0]  + conf[1][1])
    f1 = 2*(1/(1/recall + 1/precision))
    print(conf)
    print('accuracy = {:.3f}'.format( accuracy))
    print('precision = {:.3f}'.format(precision))
    print('recall = {:.3f}'.format(recall))
    print('f1-score = {:.3f}'.format(f1))
    return 


df_15 = pd.read_csv("pl_15-16.csv", encoding='cp949')
df_16 = pd.read_csv("pl_16-17.csv", encoding='cp949')
df_17 = pd.read_csv("pl_17-18.csv", encoding='cp949')
df_18 = pd.read_csv("pl_18-19.csv", encoding='cp949')
df_19 = pd.read_csv("pl_19-20.csv", encoding='cp949')
df_20 = pd.read_csv("pl_20-21.csv", encoding='cp949')


df_15.fillna(0, inplace = True)
df_16.fillna(0, inplace = True)
df_17.fillna(0, inplace = True)
df_18.fillna(0, inplace = True)
df_19.fillna(0, inplace = True)
df_20.fillna(0, inplace = True)




from sklearn.preprocessing import MinMaxScaler
MMscaler = MinMaxScaler()
X_15 = MMscaler.fit_transform(df_15.iloc[:,3:(len(df_15.columns)-1)])
X_16 = MMscaler.fit_transform(df_16.iloc[:,3:(len(df_15.columns)-1)])
X_17 = MMscaler.fit_transform(df_17.iloc[:,3:(len(df_15.columns)-1)])
X_18 = MMscaler.fit_transform(df_18.iloc[:,3:(len(df_15.columns)-1)])
X_19 = MMscaler.fit_transform(df_19.iloc[:,3:(len(df_15.columns)-1)])
X_20 = MMscaler.fit_transform(df_20.iloc[:,3:(len(df_15.columns)-1)])




X_15 = pd.DataFrame(X_15)
X_16 = pd.DataFrame(X_16)
X_17 = pd.DataFrame(X_17)
X_18 = pd.DataFrame(X_18)
X_19 = pd.DataFrame(X_19)
X_20 = pd.DataFrame(X_20)


X_15.columns = df_15.iloc[:,3:(len(df_15.columns)-1)].columns
X_16.columns = df_16.iloc[:,3:(len(df_16.columns)-1)].columns
X_17.columns = df_17.iloc[:,3:(len(df_17.columns)-1)].columns
X_18.columns = df_18.iloc[:,3:(len(df_18.columns)-1)].columns
X_19.columns = df_19.iloc[:,3:(len(df_19.columns)-1)].columns
X_20.columns = df_20.iloc[:,3:(len(df_20.columns)-1)].columns


df_15_MM = pd.concat([df_15.iloc[:,:3], X_15, df_15.iloc[:,54]], axis = 1)
df_16_MM = pd.concat([df_16.iloc[:,:3], X_16, df_16.iloc[:,54]], axis = 1)
df_17_MM = pd.concat([df_17.iloc[:,:3], X_17, df_17.iloc[:,54]], axis = 1)
df_18_MM = pd.concat([df_18.iloc[:,:3], X_18, df_18.iloc[:,54]], axis = 1)
df_19_MM = pd.concat([df_19.iloc[:,:3], X_19, df_19.iloc[:,54]], axis = 1)
df_20_MM = pd.concat([df_20.iloc[:,:3], X_20, df_20.iloc[:,54]], axis = 1)

########################################################################







#df = pd.concat([df_15_MM, df_16_MM, df_17_MM, df_18_MM, df_19_MM, df_20_MM], axis = 0 )

#15,16,17,18 시즌 train 19,20 시즌 test
df_train = pd.concat([df_15_MM, df_16_MM, df_17_MM, df_18_MM], axis = 0)
df_test = pd.concat([df_19_MM, df_20_MM])

df_FW_train = df_train[df_train['Position'] == 'Forward']
df_MF_train = df_train[df_train['Position'] == 'Midfielder']
df_DF_train = df_train[df_train['Position'] == 'Defender']
df_GK_train = df_train[df_train['Position'] == 'Goalkeeper']

df_FW_test = df_test[df_test['Position'] == 'Forward']
df_MF_test = df_test[df_test['Position'] == 'Midfielder']
df_DF_test = df_test[df_test['Position'] == 'Defender']
df_GK_test = df_test[df_test['Position'] == 'Goalkeeper']


df_FW_train=df_FW_train.iloc[:,[1,3,6,9,10,11,12,22,23,24,25,26,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_MF_train=df_MF_train.iloc[:,[1,3,6,7,9,10,11,12,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_DF_train=df_DF_train.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,54]]
df_GK_train=df_GK_train.iloc[:,[1,3,4,5,20,21,22,23,24,29,30,31,32,34,46,47,48,49,50,51,52,53,54]]

df_FW_test=df_FW_test.iloc[:,[1,3,6,9,10,11,12,22,23,24,25,26,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_MF_test=df_MF_test.iloc[:,[1,3,6,7,9,10,11,12,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,54]]
df_DF_test=df_DF_test.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,54]]
df_GK_test=df_GK_test.iloc[:,[1,3,4,5,20,21,22,23,24,29,30,31,32,34,46,47,48,49,50,51,52,53,54]]


df_FW_train.reset_index(inplace = True, drop = True)
df_MF_train.reset_index(inplace = True, drop = True)
df_DF_train.reset_index(inplace = True, drop = True)
df_GK_train.reset_index(inplace = True, drop = True)

df_FW_test.reset_index(inplace = True, drop = True)
df_MF_test.reset_index(inplace = True, drop = True)
df_DF_test.reset_index(inplace = True, drop = True)
df_GK_test.reset_index(inplace = True, drop = True)


#X = df_DF_train.iloc[:,1:(len(df_DF_train.columns)-1)]
#y = df_DF_test.iloc[:,(len(df_DF_test.columns)-1)]
#X = df_DF.iloc[:,1:(len(df_DF.columns)-1)]
#y= df_DF.iloc[:,(len(df_DF.columns)-1)]



X_train = df_FW_train.iloc[:,1:(len(df_FW_train.columns)-1)]
y_train = df_FW_train.iloc[:,(len(df_FW_test.columns)-1)]

X_test = df_FW_test.iloc[:,1:(len(df_FW_train.columns)-1)]
y_test = df_FW_test.iloc[:,(len(df_FW_test.columns)-1)]


from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.svm import SVC
from sklearn import tree


penalty = 'none'
C = 10

Log_reg = LogisticRegression(penalty= penalty , C = C , solver = 'saga', random_state=1)
Log_reg.fit(X_train, y_train)
conf_train_Log = confusion_matrix(y_train,Log_reg.predict(X_train))
conf_test_Log = confusion_matrix(y_test,Log_reg.predict(X_test))

accuracy_train = (conf_train_Log[0][0] + conf_train_Log[1][1]) / (conf_train_Log[0][0] + conf_train_Log[1][0] + conf_train_Log[0][1] + conf_train_Log[1][1])
precision_train = (conf_train_Log[1][1] ) / (conf_train_Log[1][1] + conf_train_Log[0][1] )
recall_train = (conf_train_Log[1][1]) / (conf_train_Log[1][0]  + conf_train_Log[1][1])



accuracy_test = (conf_test_Log[0][0] + conf_test_Log[1][1]) / (conf_test_Log[0][0] + conf_test_Log[1][0] + conf_test_Log[0][1] + conf_test_Log[1][1])
precision_test = (conf_test_Log[1][1] ) / (conf_test_Log[1][1] + conf_test_Log[0][1] )
recall_test = (conf_test_Log[1][1]) / (conf_test_Log[1][0]  + conf_test_Log[1][1])




print('')
print('confusion_matrix_Log_train')
print('penalty = ' , penalty)
print('C = ', C)
confu(conf_train_Log)
print('ROC AUC = ',round(roc_auc_score(y_train,Log_reg.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_Log_test')
print('penalty = ' , penalty)
print('C = ', C)
confu(conf_test_Log)
print('ROC AUC = ',round(roc_auc_score(y_test,Log_reg.predict(X_test)),2))
print('')


Log_reg.coef_

n_neighbors = 3
p = 2


knn = KNeighborsClassifier(n_neighbors = n_neighbors, p = p)   # n_neighbors = 3 , 5 , 7  p = 5 ~ 10
knn.fit(X_train, y_train)
conf_train_knn = confusion_matrix(y_train, knn.predict(X_train))
conf_test_knn = confusion_matrix(y_test, knn.predict(X_test))

accuracy_train = (conf_train_knn[0][0] + conf_train_knn[1][1]) / (conf_train_knn[0][0] + conf_train_knn[1][0] + conf_train_knn[0][1] + conf_train_knn[1][1])
precision_train = (conf_train_knn[1][1] ) / (conf_train_knn[1][1] + conf_train_knn[0][1] )
recall_train = (conf_train_knn[1][1]) / (conf_train_knn[1][0]  + conf_train_knn[1][1])


accuracy_test = (conf_test_knn[0][0] + conf_test_knn[1][1]) / (conf_test_knn[0][0] + conf_test_knn[1][0] + conf_test_knn[0][1] + conf_test_knn[1][1])
precision_test = (conf_test_knn[1][1] ) / (conf_test_knn[1][1] + conf_test_knn[0][1] )
recall_test = (conf_test_knn[1][1]) / (conf_test_knn[1][0]  + conf_test_knn[1][1])



print('')
print('confusion_matrix_KNN_train')
print('n_neighbors = ',  n_neighbors)
print('p = ',  p)
confu(conf_train_knn)
print('ROC AUC = ', round(roc_auc_score(y_train, knn.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_KNN_test')
print('n_neighbors = ', n_neighbors)
print('p = ', p)
confu(conf_test_knn)
print('ROC AUC = ', round(roc_auc_score(y_test, knn.predict(X_test)),2))
print('')



kernel = 'linear'
C = 10



svm = SVC(kernel = kernel, C = C, random_state=1)  # kernel : linear, rbf, poly, sigmoid  ,  C  = 0.1 ~ 10 
svm.fit(X_train, y_train)
conf_train_svm = confusion_matrix(y_train, svm.predict(X_train))
conf_test_svm = confusion_matrix(y_test, svm.predict(X_test))

accuracy_train = (conf_train_svm[0][0] + conf_train_svm[1][1]) / (conf_train_svm[0][0] + conf_train_svm[1][0] + conf_train_svm[0][1] + conf_train_svm[1][1])
precision_train = (conf_train_svm[1][1] ) / (conf_train_svm[1][1] + conf_train_svm[0][1] )
recall_train = (conf_train_svm[1][1]) / (conf_train_svm[1][0]  + conf_train_svm[1][1])



accuracy_test = (conf_test_svm[0][0] + conf_test_svm[1][1]) / (conf_test_svm[0][0] + conf_test_svm[1][0] + conf_test_svm[0][1] + conf_test_svm[1][1])
precision_test = (conf_test_svm[1][1] ) / (conf_test_svm[1][1] + conf_test_svm[0][1] )
recall_test = (conf_test_svm[1][1]) / (conf_test_svm[1][0]  + conf_test_svm[1][1])




print('')
print('confusion_matrix_SVM_train')
print('kernel = ', kernel )
print('C = ', C )
confu(conf_train_svm)
print('ROC AUC = ', round(roc_auc_score(y_train, svm.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_SVM_test')
print('kernel = ', kernel )
print('C = ', C)
confu(conf_test_svm)
print('ROC AUC = ', round(roc_auc_score(y_test, svm.predict(X_test)),2))
print('')




criterion = 'entropy'
max_depth = 7



dtc = tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, random_state = 1)   # criterion = gini, entropy , max_depth = 3 ~ 10 정도?
dtc.fit(X_train, y_train)
conf_train_dtc = confusion_matrix(y_train, dtc.predict(X_train))
conf_test_dtc = confusion_matrix(y_test, dtc.predict(X_test))


accuracy_train = (conf_train_dtc[0][0] + conf_train_dtc[1][1]) / (conf_train_dtc[0][0] + conf_train_dtc[1][0] + conf_train_dtc[0][1] + conf_train_dtc[1][1])
precision_train = (conf_train_dtc[1][1] ) / (conf_train_dtc[1][1] + conf_train_dtc[0][1] )
recall_train = (conf_train_dtc[1][1]) / (conf_train_dtc[1][0]  + conf_train_dtc[1][1])



accuracy_test = (conf_test_dtc[0][0] + conf_test_dtc[1][1]) / (conf_test_dtc[0][0] + conf_test_dtc[1][0] + conf_test_dtc[0][1] + conf_test_dtc[1][1])
precision_test = (conf_test_dtc[1][1] ) / (conf_test_dtc[1][1] + conf_test_dtc[0][1] )
recall_test = (conf_test_dtc[1][1]) / (conf_test_dtc[1][0]  + conf_test_dtc[1][1])


print('')
print('confusion_matrix_DT_train')
print('criterion = ', criterion )
print('max_depth = ', max_depth)
confu(conf_train_dtc)
print('ROC AUC = ',round(roc_auc_score(y_train, dtc.predict(X_train)),2))
print('')

print('')
print('confusion_matrix_DT_test')
print('criterion = ', criterion)
print('max_depth = ', max_depth)
confu(conf_test_dtc)
print('ROC AUC = ',round(roc_auc_score(y_test, dtc.predict(X_test)),2))
print('')








#ENSEMBLE _ Voting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_cl = LogisticRegression(penalty='l2', C=0.1)
rf_cl = RandomForestClassifier(max_depth = 3, n_estimators = 21)
svm_cl = SVC()
voting_cl = VotingClassifier(estimators = [('lr', log_cl), ('rf', rf_cl), ('svc', svm_cl)], voting = 'hard')
voting_cl.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

for cl in ([log_cl, rf_cl, svm_cl, voting_cl]):
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    print(cl.__class__.__name__, accuracy_score(y_test, y_pred))
    print(cl.__class__.__name__, precision_score(y_test, y_pred, pos_label=1 ))
    print(cl.__class__.__name__, recall_score(y_test, y_pred, pos_label=1))
    print(cl.__class__.__name__, f1_score(y_test, y_pred, pos_label=1))

'''
rf_cl = RandomForestClassifier(max_depth = 3, n_estimators = 21)
rf_cl.fit(X_train, y_train)
y_pred = cl.predict(X_test)
print(cl.__class__.__name__, accuracy_score(y_test, y_pred))
print(cl.__class__.__name__, precision_score(y_test, y_pred))
print(cl.__class__.__name__, recall_score(y_test, y_pred))
print(cl.__class__.__name__, f1_score(y_test, y_pred))

confusion_matrix(y_train,rf_cl.predict(X_train))
confusion_matrix(y_test,rf_cl.predict(X_test))
'''

log_cl = LogisticRegression(penalty='l2', C=0.1)
rf_cl = RandomForestClassifier(max_depth = 3, n_estimators = 21)
svm_cl = SVC(kernel='linear', C=1.0, random_state=0, probability=True)
voting_cl = VotingClassifier(estimators = [('lr', log_cl), ('rf', rf_cl), ('svc', svm_cl)], voting = 'soft')
voting_cl.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

for cl in ([log_cl, rf_cl, svm_cl, voting_cl]):
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    print(cl.__class__.__name__, accuracy_score(y_test, y_pred))
    print(cl.__class__.__name__, precision_score(y_test, y_pred, pos_label=1 ))
    print(cl.__class__.__name__, recall_score(y_test, y_pred, pos_label=1))
    print(cl.__class__.__name__, f1_score(y_test, y_pred, pos_label=1))
    
    
    
    
    
    
#ENSEMBLE_Bagging
#악한 학습기를 순차적으로 학습을 하되, 이전 학습에 대하여 잘멋 예측된 데이터에 가중치를 부여해 오차를 보완해 나가는 방식

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_cl = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, max_samples = 100, bootstrap = True)

bag_cl.fit(X_train, y_train)
y_pred = bag_cl.predict(X_test)


print('\n')
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))



#ENSEMBLE_AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_t = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), n_estimators = 500, random_state = 1)
ada_t.fit(X_train, y_train)
y_train_pred = ada_t.predict(X_train)
y_test_pred = ada_t.predict(X_test)

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))
print(roc_auc_score(y_train, y_train_pred))


print('\n')
print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))
print(roc_auc_score(y_test, y_test_pred))




#ENSEMBLE_Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
gbcl = GradientBoostingClassifier(n_estimators = 100, max_depth = 2)
gbcl.fit(X_train, y_train)

accuracies = [accuracy_score(y_test, y_pred) for y_pred in gbcl.staged_predict(X_test)]
best_n_estimator = np.argmax(accuracies)
gbcl_best = GradientBoostingClassifier(max_depth = 2, n_estimators = best_n_estimator+1)
gbcl_best.fit(X_train, y_train)
y_train_pred = gbcl_best.predict(X_train)
y_test_pred = gbcl_best.predict(X_test)

print(accuracy_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))
print(roc_auc_score(y_train, y_train_pred))
print('\n')
print(accuracy_score(y_test, y_test_pred))
print(precision_score(y_test, y_test_pred))
print(recall_score(y_test, y_test_pred))
print(f1_score(y_test, y_test_pred))
print(roc_auc_score(y_test, y_test_pred))

print("best_n_estimator",best_n_estimator)






#ENSEMBLE_XGBOOST
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
xgb_wrapper.fit(X_train, y_train)
y_pred = xgb_wrapper.predict(X_test)
# 예측 결과 확인

conf_train_xgb = confusion_matrix(y_train, xgb_wrapper.predict(X_train))
conf_test_xgb = confusion_matrix(y_test, xgb_wrapper.predict(X_test))

print(conf_train_xgb,'\n',conf_test_xgb)

print(accuracy_score(y_train, xgb_wrapper.predict(X_train)))
print(precision_score(y_train, xgb_wrapper.predict(X_train)))
print(recall_score(y_train, xgb_wrapper.predict(X_train)))
print(f1_score(y_train, xgb_wrapper.predict(X_train)))
print(roc_auc_score(y_train, xgb_wrapper.predict(X_train)))

print('\n')

print(accuracy_score(y_test, xgb_wrapper.predict(X_test)))
print(precision_score(y_test, xgb_wrapper.predict(X_test)))
print(recall_score(y_test, xgb_wrapper.predict(X_test)))
print(f1_score(y_test, xgb_wrapper.predict(X_test)))
print(roc_auc_score(y_test, xgb_wrapper.predict(X_test)))



#ENSEMBLE_Catboost
from catboost import Pool, CatBoostClassifier

train_dataset = Pool(data=X_train, label = y_train)
eval_dataset = Pool(data = X_test, label = y_test)

model = CatBoostClassifier(l2_leaf_reg= 0.01, iterations = 10, depth = 2, eval_metric = 'Accuracy')
model.fit(train_dataset, use_best_model=True, eval_set = eval_dataset)


































