# creating my first module:

# libraries
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Explore(file, column_names=None, title_line_number=100, head_line_number=20):
    #df = pd.read_csv(file, header=None, names=column_names)
    df = pd.read_csv(file);print(title_line_number*'*')
    print('The dataset has been loaded from | {} | Successfully'.format(file))
    print(title_line_number*'*'+'\n')
    print(df.head());print(title_line_number*'*'+'\n');print('\n'+title_line_number*'=')
    print('The data set has {} number of records, and {} number of columns'.format(df.shape[0],df.shape[1]))
    print(title_line_number*'*'+'\n');print('\n'+title_line_number*'=')
    print('The Datatypes are:');print(head_line_number*'-');
    print(df.dtypes);print(title_line_number*'*'+'\n');print('\n'+title_line_number*'=')    
    print('Other info:');print(head_line_number*'-');
    print(df.info());print(title_line_number*'*'+'\n');print('\n'+title_line_number*'=')
    print('Statistical Summary:');print(head_line_number*'-');
    print(df.describe());print(title_line_number*'*'+'\n');print('\n'+title_line_number*'=')
    return df

def title(string, icon='-'):
    print(string.center(100,icon))

def setJupyterNotebook():
    import pandas as pd;import numpy as np
    np.set_printoptions(precision=3)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    np.random.seed(8)
    import warnings
    warnings.filterwarnings('ignore')


def Split(df,target='target',test_size=0.3,random_state=8):
    '''
    input: pandas dataframe, target='target', test_size=0.3,random_state=8
    output: tuple of X_train, X_test, y_train, y_test
    '''
    X,y = df.drop([target], axis=1),df[target]
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def OHE(data,non_features,cat_features=None): # Use later OneHotEncoder of sklearn and fit_transform(X_train) and transform (X_test)
    X_train, X_test, y_train, y_test = data
    if cat_features is None:
        cat_features = [col for col in X_train.select_dtypes('object').columns if col not in non_features]
    X_train_cat, X_test_cat = tuple([pd.concat([pd.get_dummies(X_cat[col],drop_first=False,prefix=col,prefix_sep='_',)\
               for col in cat_features],axis=1) for X_cat in data[:2]])
    X_train = pd.concat([X_train,X_train_cat],axis=1).drop(cat_features,axis=1)
    X_test = pd.concat([X_test,X_test_cat],axis=1).drop(cat_features,axis=1)
    OHE_features = list(X_train_cat.columns)
    return (X_train, X_test, y_train, y_test), OHE_features

def Balance(data):
    '''
    input: data = tuple of X_train, X_test, y_train, y_test
           target='target' # column name of the target variable
    output: data = the balanced version of data
    => FUNCTION DOES BALANCING ONLY ON TRAIN DATASET
    '''
    X_train, X_test, y_train, y_test = data
    target=y_train.name #if else 'target'
    print('Checking Imbalance');print(y_train.value_counts(normalize=True))
    Input = input('Do You Want to Treat Data?\nPress "y" or "n" \n')
    if Input.strip() == "y":
        print('Treating Imbalance on Train Data')
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import NearMiss
        SM = SMOTE(random_state=8, ratio=1.0)
        X_train_SM, y_train_SM = SM.fit_sample(X_train, y_train)
        X_train_SM = pd.DataFrame(X_train_SM, columns=X_train.columns)
        y_train_SM = pd.Series(y_train_SM,name=target);
        print('After Balancing')
        print(y_train_SM.value_counts(normalize=True));
        print('*',"*");plt.figure(figsize=(8,3));
        plt.subplot(1,2,1);sns.countplot(y_train);plt.title('before Imbalance');
        plt.subplot(1,2,2);sns.countplot(y_train_SM);plt.title('after Imbalance Treatment');plt.show()
        data = X_train_SM,X_test, y_train_SM, y_test
    
    elif Input.strip()=='n':
        sns.countplot(y_train);plt.print('BEFORE');
        data = data
    
    return data


def SetIndex(data, index = 'ID'):
    '''
    setting index  before puting to ML algorithms and manual label encoding of y_train and y_test
    '''
    X_train, X_test, y_train, y_test = data
    y_train = y_train.map({'Yes':1,'No':0}); y_test = y_test.map({'Yes':1,'No':0})

    try:X_train = X_train.set_index(index)
    except:X_train=X_train
    y_train.index=X_train.index
    try:X_test = X_test.set_index(index)
    except:X_test=X_test
    y_test.index=X_test.index
    data = X_train, X_test, y_train, y_test
    return data

def FeatureScale(data,OHE_features,scaler='MinMaxScaler'):
    '''
        Feature Scaling only numerical_feaures. and not on OHE features
        input   data = X_train, X_test, y_train, y_test
                OHE_features = list of One Encoded categorical feature columns
                scaler = either 'StandardScaler' or 'MinMaxScaler'
        output data = X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = data
    X_train_num = X_train[[col for col in X_train.columns if col not in OHE_features]]
    X_train_cat = X_train[[col for col in X_train.columns if col in OHE_features]]
    X_test_num = X_test[[col for col in X_test.columns if col not in OHE_features]]
    X_test_cat = X_test[[col for col in X_test.columns if col in OHE_features]]

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scalers = {'StandardScaler':StandardScaler(),'MinMaxScaler':MinMaxScaler()}
    sc = scalers[scaler]
    print('Applying',scaler)
    sc_X_train= pd.DataFrame(sc.fit_transform(X_train_num),columns=X_train_num.columns,index=X_train_num.index)
    sc_X_test = pd.DataFrame(sc.transform(X_test_num),columns=X_test_num.columns,index=X_test_num.index)

    X_train_scale = pd.concat([sc_X_train,X_train_cat],axis=1)
    X_test_scale = pd.concat([sc_X_test,X_test_cat],axis=1)
    
    data = X_train_scale, X_test_scale, y_train, y_test 
    return data

def FeatureScaleAll(data,scaler='MinMaxScaler'):
    '''
    FeaturesScaling Both on OHE columns and numerical columns
    input   data = X_train, X_test, y_train, y_test
            scaler = either 'StandardScaler' or 'MinMaxScaler'
    output data = X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = data
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scalers = {'StandardScaler':StandardScaler(),'MinMaxScaler':MinMaxScaler()}
    sc = scalers[scaler]
    print('Applying',scaler)
    sc_X_train= pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
    sc_X_test = pd.DataFrame(sc.transform(X_test),columns=X_test.columns,index=X_test.index)
    data = sc_X_train, sc_X_test, y_train, y_test 
    return data


#importing Algorithms


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.svm import SVC

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier # inspired by LGBM

from lightgbm import LGBMClassifier

def ClassificationModelDictionary():
    LR = dict(name ='LogisticRegression',model = LogisticRegression(),
         parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
         best_parameters = {},
         cv_params={'penalty': ['l1', 'l2'],'random_state':[0,8]})
    DT = dict(name ='DecisionTreeClassifier',model = DecisionTreeClassifier(),
         parameters = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],
                       'max_depth': [None,2,3,4,5,6,7,8,9,10], 'max_features': ['auto', 'log2',None],
                       'random_state': [8],'min_samples_leaf' : [1,2,3,4,5]},
         best_parameters = {},
         cv_params= {'criterion': ['gini', 'entropy'],'splitter': ['best'],
                    'max_features': ['auto', 'log2', None],'random_state': [0,8]}
)

    KNN= dict(name = 'KNeighborsClassifier',
              model = KNeighborsClassifier(),
              parameters = {'n_neighbors': [i for i in range(1,25)],
                            'p':[1,2]}, # 1=manhattan, 2, euclidean
             best_parameters = {},
             cv_params={'priors': [None], 'var_smoothing': [1e-09]})

    GNB= dict(name = 'GaussianNB',
              model = GaussianNB(),
              parameters = {'priors':[None,],'var_smoothing':[1e-09,]},
              best_parameters = {},
              cv_params={'priors': [None], 'var_smoothing': [1e-09]})
    BNB= dict(name = 'BernoulliNB',
              model = BernoulliNB(),
              parameters = {'alpha':[1.0,],
                            'binarize':[0.0,],
                            'fit_prior':[True,False],
                            'class_prior':[None]},
              best_parameters = {},
              cv_params={'alpha': [1.0],'binarize': [0.0],
                        'fit_prior': [True, False],'class_prior': [None]})

    RF= dict(name = 'RandomForestClassifier',
              model = RandomForestClassifier(),
              parameters = {'max_depth': [2, 3, 4],
                            'bootstrap': [True, False],
                            'max_features': ['auto', 'sqrt', 'log2', None],
                            'criterion': ['gini', 'entropy'],
                            'random_state': [8]},
             best_parameters = {},
             cv_params= {'max_depth': [2, 3, 4],'bootstrap': [True, False],
                         'max_features': ['auto', 'sqrt', 'log2', None],'criterion': ['gini', 'entropy'],
                         'random_state': [8]})
    SVM= dict(name = 'SVC',
              model = SVC(),
              parameters = {'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
                            'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
                            #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
                                    },
              best_parameters = {},
              cv_params={'C': [1, 10, 100, 500, 1000],'kernel': ['rbf'],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
)


    BAG_params={'base_estimator': [DecisionTreeClassifier(),
                                   DecisionTreeClassifier(max_depth=2),
                                   DecisionTreeClassifier(max_depth=4),
                                   BernoulliNB(),
                                   LogisticRegression(penalty='l1'),
                                   LogisticRegression(penalty='l2'),
                                   ], #GaussianNB(),],
                'n_estimators': [10,], 
                'max_samples': [1.0], 'max_features': [1.0], 
                'bootstrap': [True,], 'bootstrap_features': [False], 
                'oob_score': [False], #'warm_start': [False], 
                'n_jobs': [None], 'random_state': [8], 'verbose': [0]}

    BAG= dict(name = 'BaggingClassifier',
              model= BaggingClassifier(),
              parameters = BAG_params,
              best_parameters = {},
              cv_params={'base_estimator': [DecisionTreeClassifier(criterion='gini'),
                                            DecisionTreeClassifier(criterion='entropy'),
                                            BernoulliNB(),
                                            LogisticRegression(penalty='l1'),
                                            LogisticRegression(penalty='l2')],
                            'bootstrap': [True],
                            'random_state':[0,8] }
             )

    GB = dict(name = 'GradientBoostingClassifier',
              model = GradientBoostingClassifier(),
              parameters = {
                  'loss':['deviance','exponential'],
                  'learning_rate':[0.1,0.01,1.0],
                  'n_estimators':[100,200,25,50,75],
                  'subsample':[1.0,0.75,0.5,0.25,0.01], # < 1.0 leads to reduction of variance and increase in bias
                                    #  < 1.0 results in Stochastic Gradient Boosting
                  'random_state':[8],
                  #'ccp_alpha': [0.0,0.0001,0.001,0.01,0.1,1.0]# only in version 0.22
                                  #cost-complexity pruning algorithm to prune tree to avoid over fitting
                  #'min_samples_split':[2,3,4],
                  #'min_samples_leaf':[1,2,3],
                  #'min_weight_fraction_leaf':[0],
                  #'max_depth':[3,4,5],
                  #'min_impurity_decrease':[0],
                  #'init':[None],
                  #'max_features':[None],
                  #'verbose':[0],

              },
              best_parameters = {},
              cv_params = {'loss': ['deviance', 'exponential'],
                            'n_estimators': [100],'random_state': [0,8]}
              )
    ADA= dict(name = 'AdaBoostClassifier',
              model = AdaBoostClassifier(),
              parameters = {'base_estimator':[DecisionTreeClassifier(max_depth=1),
                                                 DecisionTreeClassifier(max_depth=2),
                                                 DecisionTreeClassifier(max_depth=3),
                                                 DecisionTreeClassifier(max_depth=4),
                                                 BernoulliNB(),
                                                 #GaussianNB(),
                                             ],
                            'n_estimators':[25,50,75,100],# ,100
                            'learning_rate':[1.0,0.1],
                            #'alogorithm':['SAMME', 'SAMME.R'],
                            'random_state':[8],
                           },
              best_parameters = {},
              cv_params = {'base_estimator': [None,DecisionTreeClassifier(criterion='gini'),
                                                    DecisionTreeClassifier(criterion='entropy'),
                                                    BernoulliNB(),
                                                    LogisticRegression(penalty='l1'),
                                                    LogisticRegression(penalty='l2')],
                            'random_state':[0,8] })

    XGB_params = {'max_depth': [3],'learning_rate': [0.1],'n_estimators': [100,],#50,150,200],
                  'verbosity': [1],'objective': ['binary:logistic'],
                  'booster': ['gbtree', 'gblinear','dart'], # IMPORTANT
                  'tree_method': ['auto', 'exact', 'approx', 'hist'],#, 'gpu_hist' # IMPORTANT
                  'n_jobs': [1],'gamma': [0],
                  'min_child_weight': [1],'max_delta_step': [0],
                  'subsample': [1],
                  'colsample_bytree': [1],'colsample_bylevel': [1],'colsample_bynode': [1],
                  'reg_alpha': [0],'reg_lambda': [1],'scale_pos_weight': [1],'base_score': [0.5],
                  'random_state': [8],'missing': [None]}

    XGB= dict(name = 'XGBClassifier',
              model= XGBClassifier(),
              parameters = XGB_params,
              best_parameters = {},
              cv_params = {'tree_method': ['auto', 'exact', 'approx', 'hist'],
                              'booster': ['gbtree', 'gblinear', 'dart'],
                              'random_state':[0,8]}
             )

    LBGM_params={'boosting_type': ['gbdt','goss'], # ,'dart','rf'
                 'num_leaves': [31], 'max_depth': [-1], 'learning_rate': [0.1], 
                 'n_estimators': [100], 'subsample_for_bin': [200000], 'objective': [None],
                 'class_weight': [None], 'min_split_gain': [0.0], 'min_child_weight': [0.001],
                 'min_child_samples': [20], 'subsample': [1.0], 'subsample_freq': [0], 
                 'colsample_bytree': [1.0], 'reg_alpha': [0.0], 'reg_lambda': [0.0], 
                 'random_state': [8], 'n_jobs': [-1], 'silent': [True], 'importance_type': ['split']}

    LGBM= dict(name = 'LGBMClassifier',
              model= LGBMClassifier(),
              parameters = LBGM_params,
              best_parameters = {},
              cv_params = {'boosting_type': ['gbdt', 'goss'],
                               'random_state':[0,8]}
             )

    HGB_params={'loss': ['auto','binary_crossentropy',], # 'categorical_crossentropy'
                'learning_rate': [0.1], 'max_iter': [100], 'max_leaf_nodes': [31],
                'max_depth': [None], 'min_samples_leaf': [20], 
                'l2_regularization': [0,1,2], # for no-regulaiziation, 1 regulztn
                'max_bins': [255], 
                #'warm_start': [False],
                'scoring': [None], 'validation_fraction': [0.1],
                'n_iter_no_change': [None], 'tol': [1e-07], 'verbose': [0],
                'random_state': [8]}

    HGB= dict(name = 'HistGradientBoostingClassifier',
              model= HistGradientBoostingClassifier(),
              parameters = HGB_params,
              best_parameters = {},
              cv_params = {'loss': ['auto', 'binary_crossentropy'],
                                                'l2_regularization': [0, 1, 2],
                                                'random_state':[0,8]}
             )

    models = {i:mod for i,mod in enumerate([LR,DT,KNN,GNB,BNB,RF,SVM,BAG,GB,ADA,XGB,LGBM,HGB],start=1)}
    return models


def MODEL(model_dict,data,phase='',scores=None,use_params=False):
    '''
    input =>
        model_dict  : Each Individual Models in a dictionary, 
        data        : (X_train,X_test,y_train,y_test) 
        phase       : '' (default) [options like base, final, HPO, etc...]
        scores      : None (default) scores id data frame with cols: 'Model','Phase','AUC_ROC','TrainingAccuracy
                            'TestingAccuracy','Recall','Precision','F1_Score','FalsePositives','FalseNegatives'
        use_params  : False (default) -- uses best_parameters from model_dict
    output =>
        tuple of dictionary{model_name:model} and scores (DataFrame)
    '''
    X_train, X_test, y_train, y_test = data
    if scores is None: scores=pd.DataFrame(columns=['Model','Phase','AUC_ROC','TrainingAccuracy',
                                             'TestingAccuracy','Recall','Precision','F1_Score',
                                             'FalsePositives','FalseNegatives'])
    model = model_dict['model']
    if use_params:model.set_params(**model_dict['best_parameters'])
    algorithm_name = model_dict['name']
    model_name = algorithm_name+phase
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score,confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    record = [{'Model':algorithm_name,'Phase':phase,
               'AUC_ROC':roc_auc_score(y_test,y_pred),
               'TrainingAccuracy':accuracy_score(y_train,model.predict(X_train)),
               'TestingAccuracy':accuracy_score(y_test,y_pred),
               'Recall':recall_score(y_test,y_pred),
               'Precision':precision_score(y_test,y_pred),
               'F1_Score':f1_score(y_test,y_pred),
               'FalsePositives':fp,
               'FalseNegatives':fn,
              }]
    scores =scores.append(pd.DataFrame(record),sort=False)
    return {model_name:model}, scores


def RunAll(models,data,phase='',scores=None,trained_models = {},use_params=False):
    '''
    Run All Alogithms:
    input =>
        models      : a dictionary of All Models 
        data        : (X_train,X_test,y_train,y_test) 
        phase       : '' (default) [options like base, final, HPO, etc...]
        scores      : None (default) scores id data frame with cols: 'Model','Phase','AUC_ROC','TrainingAccuracy
                            'TestingAccuracy','Recall','Precision','F1_Score','FalsePositives','FalseNegatives'
        trained_models : {} (default)     -- a dictionary of all trained models and it's unique names
        use_params  : False (default) -- uses best_parameters from model_dict
    output =>
        tuple of trained_models (dictionary) and scores (DataFrame)
    '''
    if scores is None: scores=pd.DataFrame(columns=['Model','Phase','AUC_ROC','TrainingAccuracy',
                                             'TestingAccuracy','Recall','Precision','F1_Score',
                                                   'FalsePositives','FalseNegatives'])
    for i in range(1,len(models)+1):
        trained_model, scores = MODEL(model_dict=models[i],data=data,phase=phase,scores=scores,use_params=use_params)
        trained_models.update(trained_model)
    return trained_models, scores


def Classify(algorithm,model,data,phase='',scores=None):
    '''
    input: algorithm=alogorithm Name,model=alogorithm with set params,data,phase='',scores=None
    output: scores dataframe
    '''
    X_train, X_test, y_train, y_test = data
    if scores is None: scores=pd.DataFrame(columns=['Model','Phase','AUC_ROC','TrainingAccuracy',
                                             'TestingAccuracy','Recall','Precision','F1_Score',
                                             'FalsePositives','FalseNegatives'])
    model_name = algorithm+phase
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score,confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    record = [{'Model':algorithm,'Phase':phase,
               'AUC_ROC':roc_auc_score(y_test,y_pred),
               'TrainingAccuracy':accuracy_score(y_train,model.predict(X_train)),
               'TestingAccuracy':accuracy_score(y_test,y_pred),
               'Recall':recall_score(y_test,y_pred),
               'Precision':precision_score(y_test,y_pred),
               'F1_Score':f1_score(y_test,y_pred),
               'FalsePositives':fp,
               'FalseNegatives':fn,
              }]
    scores =scores.append(pd.DataFrame(record),sort=False)
    return scores.sort_values('AUC_ROC',ascending=False)


if __name__ == '__main__':
	print('''
		=================================
		Module Written by Santo K. Thomas
		=================================
		email: santokalayil@gmail.com
		phone: +91 8891960880
		Address:
			Kalayil House
			Cheenkalthadom P.O,
			Mannarakulanji, 
			Pathanamthitta
		''')