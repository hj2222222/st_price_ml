import os
# Data Wrangling
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# Visualization
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
import seaborn as sns
# EDA
import klib
# Preprocessing & Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
# Hyperparameter Optimization
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#from bayes_opt import BayesianOptimization
# Modeling
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from itertools import combinations
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import ensemble, metrics
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
# Evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

# Utility
import os
import time
import random
import warnings; warnings.filterwarnings("ignore")
from IPython.display import Image
import pickle
from tqdm import tqdm
import platform
from itertools import combinations
from scipy.stats.mstats import gmean
from vecstack import stacking

my_predictions = {}

#마지막 데이터만 export 하기 위하여 사용함.
exportrmsedata=False

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive', 
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray', 
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]


#함수 선언부-수정 필요
def plot_predictions(name_, actual, pred):
    df = pd.DataFrame({'actual': y_dev, 'prediction': pred})
    df = df.sort_values(by='actual').reset_index(drop=True)

    plt.figure(figsize=(12, 9))
    plt.scatter(df.index, df['prediction'], marker='x', color='r')
    plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
    plt.title(name_, fontsize=15)
    plt.legend(['prediction', 'actual'], fontsize=12)

def rmse_eval(name_, actual, pred):
    global predictions
    global colors
    global exportrmsedata
    plot_predictions(name_, actual, pred)

    rmse = np.sqrt(mean_squared_error(actual, pred))
    my_predictions[name_] = rmse

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(y_value, columns=['model', 'rmse'])
    print(df)
    min_ = df['rmse'].min() - 10
    max_ = df['rmse'].max() + 10
    
    length = len(df)
    
    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['rmse'])
    
    for i, v in enumerate(df['rmse']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')
        
    plt.title('RMSE Error', fontsize=18)
    plt.xlim(min_, max_)
    if exportrmsedata==True:
        plt.savefig(f'project/rmsedatas/{indexnum}_Rmse.png')
        exportrmsedata=False
        #종료 후 초기화

def getcompanylist(address):
    company_df=pd.read_csv(address)
    #여기 절대 경로를 사용자에 맞추어 바꾸어 주세요.
    companylist=company_df["종목코드"].to_list()
    for i in range(len(companylist)):
        companylist[i]=str(companylist[i]).rjust(6,"0")#padding 함수 사용
    return companylist

companyaddress="C:\programming_practice\python\python_programming\project\listed_companies.csv"
indexnumlist=getcompanylist(companyaddress)

for indexnum in indexnumlist:
    try:
        #여기 부터는 여러 번 실행하며 데이터를 얻어야 함.
        #데이터 가져오기
        main_df=pd.read_csv(f'project/companydataexcludeUSIdx/Totaldata_{indexnum}.csv',index_col="Date")
        sup_df=pd.read_csv(f'project/companydataexcludeUSIdx/supportindexscaled_{indexnum}.csv',index_col="Date")
        df=pd.merge(main_df,sup_df,on="Date")
        df=df.drop([df.columns[0],"ma_5","ma_20","ma_60","ma_120"],axis=1)
        X = df.copy()
        X = X.drop('Close',axis = 'columns')
        y = df['Close']
        y = pd.DataFrame(y)
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, shuffle =False, random_state=0)
        rf = RandomForestRegressor(random_state=0, n_jobs=4)
        gbm = GradientBoostingRegressor(random_state=0)
        lgbm = lightgbm.LGBMRegressor(random_state=0 ,n_jobs=4)
        xgb = XGBRegressor(
            max_depth=15,
            n_estimators=150,
            min_child_weight=300, 
            colsample_bytree=0.8, 
            subsample=0.8, 
            eta=0.3,    
            seed=42)
        lr = LinearRegression(n_jobs=4)
        cb = CatBoostRegressor(random_state=0 )
        elasticnet = ElasticNet(alpha=0.5, l1_ratio=0.2)
        ridge = Ridge(alpha=1)
        lasso = Lasso()
        ard = ARDRegression()
        bayesian = BayesianRidge()
        models = [rf, gbm, lgbm, xgb, lr , ridge,lasso,ard, bayesian,elasticnet]
        X_train= X_train.reset_index(drop=True)
        X_dev= X_dev.reset_index(drop=True)
        y_dev = y_dev.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_dev = y_dev.values
        y_train = y_train.values
        y_dev = np.array(y_dev).flatten().tolist()
        y_train = np.array(y_train).flatten().tolist()
        ts = time.time()
        #XGBRegressor 모델
        model = XGBRegressor(
            max_depth=15,
            n_estimators=150,
            min_child_weight=300, 
            colsample_bytree=0.8, 
            subsample=0.8, 
            eta=0.3,    
            seed=42)
        model.fit(
            X_train, 
            y_train, 
            eval_metric='rmse', 
            eval_set=[(X_train, y_train), (X_dev, y_dev)], 
            verbose=True,
            early_stopping_rounds = 10)
        time.time() - ts
        #RandomForest 모델
        rf_model = RandomForestRegressor(n_estimators=1000, max_depth=9, random_state=0, n_jobs=4)
        rf_model.fit(X_train, y_train)
        rf_train_pred = rf_model.predict(X_train)
        rf_val_pred = rf_model.predict(X_dev)
        #선형 회귀(LinearRegression)모댈
        lr_model = LinearRegression(n_jobs=4)
        lr_model.fit(X_train, y_train)
        #1.ard 모델
        ard = ARDRegression(alpha_1=0.01, alpha_2=0.01, lambda_1=1e-06, lambda_2=1e-06)
        ard.fit(X_train, y_train)
        ard_pred = ard.predict(X_dev)
        rmse_eval('ard', y_dev, ard_pred)
        #2.베이지안-Ridge 모델
        bayesian = BayesianRidge()
        bayesian.fit(X_train, y_train)
        bayesian_pred = bayesian.predict(X_dev)
        rmse_eval('bayesian', y_dev, bayesian_pred)
        #3.선형 회귀 모델
        linear_reg = LinearRegression(n_jobs=4)
        linear_reg.fit(X_train, y_train)
        linear_pred = linear_reg.predict(X_dev)
        rmse_eval('LinearRegression', y_dev, linear_pred)
        #4.ridge 모델
        ridge = Ridge(alpha=1)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_dev)
        rmse_eval('Ridge(alpha=1)', y_dev, ridge_pred)
        #5.Rasso  모델
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_dev)
        rmse_eval('Lasso(alpha=0.01)', y_dev, lasso_pred)
        #6.elasticnet 모델
        elasticnet = ElasticNet(alpha=0.5, l1_ratio=0.2)
        elasticnet.fit(X_train, y_train)
        elas_pred = elasticnet.predict(X_dev)
        rmse_eval('ElasticNet(l1_ratio=0.2)', y_dev, elas_pred)
        #7.Voring-ensemble(선형,Ridge,lasso,elasticnet 4중 구성)모델
        single_models = [
            ('lr ', lr),
            ('ridge', ridge),
            ('lasso', lasso),
            ('elasticnet', elasticnet)
        ]
        voting_regressor = VotingRegressor(single_models, n_jobs=-1)
        voting_regressor.fit(X_train, y_train)
        voting_pred = voting_regressor.predict(X_dev)
        rmse_eval('Voting Ensemble', y_dev, voting_pred)
        #8.RandomForestRegressor 사용-(주의: 원래 선언 내용이 아닌 그 다음 셀에 있는 수정된 내용을 가져와서 사용하였습니다.)
        rfr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                            max_samples=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=None, oob_score=False,
                            random_state=1, verbose=0, warm_start=False)
        rfr.fit(X_train, y_train)
        rfr_pred = rfr.predict(X_dev)
        rmse_eval('RandomForest Ensemble', y_dev, rfr_pred)
        #9.RandomForest 최적화
        param_grid = {  'bootstrap': [True], 'max_depth': [5, 10, None], 
                    'max_features': ['auto', 'log2'], 
                    'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}
        rfr = RandomForestRegressor(random_state = 1)
        g_search = GridSearchCV(estimator = rfr, param_grid = param_grid, 

                                cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)
        g_search.fit(X_train, y_train)
        rfr_t = RandomForestRegressor(bootstrap = True, max_depth = g_search.best_params_.get("max_depth"),random_state =50,max_features = "auto", n_estimators = g_search.best_params_.get("n_estimators"))
            #주의: 직전 셀에 있던 최적화 내용을 직접 적용하였습니다.
        rfr_t.fit(X_train, y_train)
        rfr_t_pred = rfr_t.predict(X_dev)
        rmse_eval('RandomForest Ensemble w/ Tuning', y_dev, rfr_t_pred)
        #10.gradientboost 모델 적용
        gbr = GradientBoostingRegressor(random_state=1)
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_dev)
        rmse_eval('GradientBoost Ensemble', y_dev, gbr_pred)
        #11.gradientboost모델 최적화
        gbr_t2 = GradientBoostingRegressor(random_state=1, learning_rate=0.01, n_estimators=1000)
        gbr_t2.fit(X_train, y_train)
        gbr_t2_pred = gbr_t2.predict(X_dev)
        rmse_eval('GradientBoost Ensemble w/ tuning (lr=0.01, est=1000)', y_dev, gbr_t2_pred)
        gbr_t3 = GradientBoostingRegressor(random_state=42, learning_rate=0.01, n_estimators=1000, subsample=0.7)
        gbr_t3.fit(X_train, y_train)
        gbr_t3_pred = gbr_t3.predict(X_dev)
        rmse_eval('GradientBoost Ensemble w/ tuning (lr=0.01, est=1000, subsample=0.7)', y_dev, gbr_t3_pred)
        #12.XGBoost 모델
        xgb = XGBRegressor(random_state=1)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_dev)
        rmse_eval('XGBoost', y_dev, xgb_pred)
        #13.XGBoost 최적화
        xgb_t = XGBRegressor(random_state=1, learning_rate=0.01, n_estimators=1000, subsample=0.7, max_features=0.8, max_depth=7)
        xgb_t.fit(X_train, y_train)
        xgb_t_pred = xgb_t.predict(X_dev)
        rmse_eval('XGBoost w/ Tuning', y_dev, xgb_t_pred)
        #14.LGBM 회귀 모델
        lgbm = LGBMRegressor(random_state=1)
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_dev)
        rmse_eval('LightGBM', y_dev, lgbm_pred)
        #15.LGBM 모델 최적화
        lgbm_t = LGBMRegressor(random_state=1, learning_rate=0.01, n_estimators=1500, colsample_bytree=0.7, subsample=0.7, max_depth=9)
        lgbm_t.fit(X_train, y_train)
        lgbm_t_pred = lgbm_t.predict(X_dev)
        rmse_eval('LightGBM w/ Tuning', y_dev, lgbm_t_pred)
        #16.Staking Ensemble
        stack_models = [
            ('linear_reg',linear_reg),
            ('ard', ard),
            ('ridge', ridge)]
        stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=2)
        stack_reg.fit(X_train, y_train)
        stack_pred = stack_reg.predict(X_dev)
        rmse_eval('Stacking Ensemble', y_dev, stack_pred)
        #17.Random Search
        params = {
            'learning_rate': [0.005, 0.01, 0.03, 0.05],
            'n_estimators': [500, 1000, 2000, 3000],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'subsample': [0.7, 0.8, 0.9, 1.0],
        }
        rcv_lgbm = RandomizedSearchCV(LGBMRegressor(), params, random_state=1, cv=5, n_iter=100, scoring='neg_mean_squared_error',n_jobs=2)
        rcv_lgbm.fit(X_train, y_train)
        lgbm_best = LGBMRegressor(learning_rate=rcv_lgbm.best_params_.get("learning_rate"),
                                n_estimators=rcv_lgbm.best_params_.get("n_estimators"), 
                                subsample=rcv_lgbm.best_params_.get("subsample"), 
                                max_depth=rcv_lgbm.best_params_.get("max_depth"),
                                colsample_bytree=rcv_lgbm.best_params_.get("colsample_bytree"),n_jobs=4)
        lgbm_best_pred = lgbm_best.fit(X_train, y_train).predict(X_dev)
        rmse_eval('RandomSearch LGBM', y_dev, lgbm_best_pred)
        #18.grid search
        params = {
            'learning_rate': [0.04, 0.05, 0.06],
            'n_estimators': [800, 1000, 1200],
            'max_depth': [3, 4, 5],
            'colsample_bytree': [0.8, 0.85, 0.9],
            'subsample': [0.8, 0.85, 0.9],
        }
        grid_search = GridSearchCV(LGBMRegressor(), params, cv=5, n_jobs=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        lgbm_best = LGBMRegressor(learning_rate=grid_search.best_params_.get("learning_rate"),
                                n_estimators=grid_search.best_params_.get("n_estimators"), 
                                subsample=grid_search.best_params_.get("supsample"),
                                max_depth=grid_search.best_params_.get("max_depth"),
                                colsample_bytree=grid_search.best_params_.get("colsample_bytree"),n_jobs=4)
        lgbm_best_pred = lgbm_best.fit(X_train, y_train).predict(X_dev)
        rmse_eval('GridSearch LGBM', y_dev, lgbm_best_pred)
        #Voting ensemble-2
        single_models = [
            ('bayesian ', bayesian),
            ('ridge', ridge),
            #('rfr', rfr),
            ('ard',ard)
            
        ]
        voting_regressor = VotingRegressor(single_models, n_jobs=4)

        voting_regressor.fit(X_train, y_train)

        VotingRegressor(estimators=[('linear_reg',
                                    LinearRegression(copy_X=True, fit_intercept=True,
                                                    n_jobs=8, normalize=False)),
                                    ('ridge',
                                    Ridge(alpha=1, copy_X=True, fit_intercept=True,
                                        max_iter=None, normalize=False,
                                        random_state=None, solver='auto',
                                        tol=0.001)),
                                    ('ard',
                                    ARDRegression())],
                                                                
                                    
                        n_jobs=4, weights=None)

        voting_pred = voting_regressor.predict(X_dev)
        rmse_eval('Voting Ensemble', y_dev, voting_pred)
        #stacking ensemble:2
        stack_models = [
            ('bayesian ', bayesian),
            ('ridge', ridge),
            #('rfr', rfr),
            ('ard',ard)]
        stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=4)
        stack_reg.fit(X_train, y_train)
        stack_pred = stack_reg.predict(X_dev)
        rmse_eval('Stacking Ensemble', y_dev, stack_pred)
        #randomForest-2
        rfr = RandomForestRegressor(random_state=1,n_jobs=4)
        rfr.fit(X_train, y_train)
        rfr_pred = rfr.predict(X_dev)
        exportrmsedata=True
        rmse_eval('RandomForest Ensemble', y_dev, rfr_pred)



        #최종 주가 예측-(베이지안,ridge,ard 모델 3개의 voting ensemble 형태로 수행)
        single_models = [
            ('bayesian ', bayesian),
            ('ridge', ridge),
            #('rfr', rfr),
            ('ard',ard)
            
        ]
        model = VotingRegressor(single_models, n_jobs=6)

        model.fit(X_train, y_train)

        VotingRegressor(estimators=[('linear_reg',
                                    LinearRegression(copy_X=True, fit_intercept=True,
                                                    n_jobs=6, normalize=False)),
                                    ('ridge',
                                    Ridge(alpha=1, copy_X=True, fit_intercept=True,
                                        max_iter=None, normalize=False,
                                        random_state=None, solver='auto',
                                        tol=0.001)),
                                    ('ard',
                                    ARDRegression())],
                                                                
                                    
                        n_jobs=6, weights=None)

        y_pred = model.predict(X_dev)

        fig = plt.figure(facecolor = 'white',figsize =(20,10))
        ax = fig.add_subplot(111)
        ax.plot(y_dev,label='True')
        ax.plot(y_pred,label = 'Prediction')
        ax.legend()
        plt.savefig(f"project/predictions/prediction_{indexnum}.png")
    except:
        #indexnum이 있으나 오류가 나는 종목은 거래정지가 된 종목이다.
        pass