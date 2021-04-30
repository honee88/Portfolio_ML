# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 3. Data Modeling 
# #### 2020_10_27

# +
# data preprocessing
import pandas as pd
import numpy as np
# data visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud

# sklearn 
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score
from sklearn.metrics import f1_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# from CatBoost import CatBoostClassifier

import re
import ast # 형변환 

from pprint import pprint as pp
import warnings
warnings.filterwarnings(action='ignore')

# 텍스트 마이닝
# from eunjeon import Mecab
# import nltk
# from konlpy.tag import Okt
# from nltk import FreqDist
# from gensim.models import Word2Vec

# +
# font
from matplotlib import font_manager, rc
import platform

if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:    
# Mac 인 경우
    rc('font', family='AppleGothic')
    
matplotlib.rcParams['axes.unicode_minus'] = False   
# -

# ##  1. 데이터 불러오기

# EDA 완료한 데이터 불러오기 
df_raw = pd.read_csv('./EDA_03')
pet = df_raw.copy()
pet.info()

col_list = [ 'age', 'sexcd', 'neuteryn', 'type', 'lat', 'long',
            'blc_col', 'brown_col', 'white_col', 'grey_col', 'yellow_col',
            'kindcd', 'weight_float', 
            'noticesdt_year','noticesdt_month', 'noticesdt_day', 
            'good_behavior', 'sensitive','disease', 'accident', 'processstate_adopt', 'processstate_euthanize']
pet = pet[col_list]

# ## 2. 결손값 처리하기 np.nan

pet.isnull().sum()

pet.good_behavior = pet.good_behavior.fillna(0)
pet.sensitive = pet.sensitive.fillna(0)
pet.disease = pet.disease.fillna(0)
pet.accident = pet.accident.fillna(0)

# +
# 몸무게 이상치 처리 : 논리적으로 판단 
# 1) 강아지 견종별 표준 몸무게 참고 : 최대 ~ 82kg  : https://parkminsuk.tistory.com/101 
# 2) 고양이 표준 몸무게 :  3.6 – 4.5 kg (Adult) : 최대 5.8kg : https://m.blog.naver.com/PostView.nhn?blogId=graindea&logNo=220363109747&proxyReferer=https:%2F%2Fwww.google.com%2F

pet[(pet.type == 1.0) & (pet.weight_float > 82)]['weight_float'].count() # 55
pet[(pet.type == 0.0) & (pet.weight_float > 10)]['weight_float'].count() # 137
# -

# 일단 결측치로 입력
pet[(pet.type == 1.0) & (pet.weight_float > 82)]['weight_float'] = np.nan # 55
pet[(pet.type == 0.0) & (pet.weight_float > 10)]['weight_float'] = np.nan # 278

# +
# 결측치 처리하기 

p = SimpleImputer(missing_values=np.nan, strategy='mean')
df_filter = pet.filter(['age','weight_float'])

res = p.fit_transform(df_filter.values)
res
# -

df_mean = pd.DataFrame(res, index = df_filter.index, columns= ['age', 'weight'])
df_mean

pet = pet.drop(columns= ['age', 'weight_float'], axis=1)

pet = pd.merge(pet, df_mean, left_index=True, right_index=True)
pet.info()

# kindcd 결측치 0으로 채우기
pet.kindcd = pet.kindcd.fillna(0)

# 인코딩/정규화 하기 전 데이터 셋 1
df_pet = pet.copy()

# ## 3. encoding 

# 원핫인코딩 - kindcd 품종정보 
df_oh = pd.get_dummies(pet, columns=['kindcd'], prefix='jong')
df_oh

# +
# # 원핫인코딩 - kindcd 품종정보 

# oh_encoder = OneHotEncoder()
# x = oh_encoder.fit_transform(pet['kindcd'].values.reshape(-1, 1)).toarray()

# +
# ohdf2 = pd.DataFrame(x, columns=['kind'+str(int(i)) for i in range(x.shape[1])])
# df_oh = pd.concat([pet, ohdf2], axis=1)
# df_oh
# -

df_oh.isnull().sum().sum()

# ## 4. scaling : Standardscaler 정규화  

scaler = StandardScaler()
scaler.fit(df_oh)
pet_oh_scaled = scaler.transform(df_oh)

df_final = pd.DataFrame(data=pet_oh_scaled, columns=df_oh.columns)
df_final

# 정규화가 잘 되었는지 확인
print('features 평균 값: ', df_final.mean()) # 각 피쳐별로 나옴 
print('features 분산 값: ', df_final.var())

# +
# label 값을 0,1로 바꾸기 
# -

df_final.processstate_adopt.value_counts()

df_final.processstate_euthanize.value_counts()

df_final.processstate_adopt = df_final.processstate_adopt.apply(lambda x : '1' if x > 0 else '0')

df_final.processstate_euthanize = df_final.processstate_euthanize.apply(lambda x : '1' if x > 0 else '0')

# 인코딩/정규화 하기 전 데이터 셋 2
df_pet2 = df_final.copy()

# 데이터 프레임 2개로 테스트 
df_pet.shape, df_pet2.shape

df_pet.to_csv('./df_pet', index=False)

df_pet2.to_csv('./df_pet2', index=False)

# ## 5. test/train set 나누기 

# +
# 데이터 셋의 점수 차이가 크지 않아서 인코딩/정규화 한 데이터셋으로 진행 
# -

df_pet2 = pd.read_csv('./df_pet2')

# 피쳐, 라벨 데이터 나누기 
label_adopt = df_pet2.processstate_adopt
label_euthanize = df_pet2.processstate_euthanize

features = df_pet2.drop(columns = ['processstate_adopt', 'processstate_euthanize'], axis=1,inplace=False)

features.info()

#  train/test split 나누기 - Stratified 로 
X_train, X_test, y_train, y_test = train_test_split(features, label_adopt, test_size=0.2, stratify=label_adopt, random_state=42)


#  validation split from train split 
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ## 6. train set 시각적 탐색

# ##### 입양 여부에 따라 어떤 시기에 입양이 잘 됐나?

# +
final_df = df_pet.drop(['lat','long'], axis=1)

final_df_0 = final_df[final_df.processstate_adopt == 0]
final_df_1 = final_df[final_df.processstate_adopt == 1]

# +
gr_df = final_df_1[['noticesdt_month','sexcd']]
k1 = gr_df.groupby('noticesdt_month')
t1 = k1.count().reset_index()
t1

gr_df = final_df_0[['noticesdt_month','sexcd']]
k2 = gr_df.groupby('noticesdt_month')
t2 = k2.count().reset_index()
t2

# +
axx = plt.plot(t1.noticesdt_month, t1.sexcd, color='green')	# HTML color name
axx = plt.plot(t2.noticesdt_month, t2.sexcd, color='red')	# HTML color name

plt.title('월별 유기 및 입양 수') 
plt.xlabel('월') 
plt.ylabel('마리')
plt.legend(labels = ['입양수', '유기수'])
# plt.to_file('./')
# -

# ##### 인사이트
# ###### 월별 입양율의 편차는 없는 반면, 7 ~ 10월에 유기수가 높아짐
# ###### 휴가철이기 때문으로 판단
#

# +
# 몸무게
weight_name = [3.0, 4.0, 1.0, 2.0, 5.0, 0.5]
ax_weight = df_pet.weight.value_counts().iloc[:6].plot(kind = 'bar', color = 'peru')
ax_weight.set_xticklabels(weight_name, rotation=60)
ax_weight.set_title('몸무게 분포')
ax_weight.set_xlabel('몸무게')
ax_weight.set_ylabel('마리')
plt.show()



# -

# 성별에 따른 입양 비율
sex_name = ['M', 'F', 'Q']
ax = sns.countplot(x = 'sexcd', hue = 'processstate_adopt', data = df_pet )
ax.set_xticklabels(sex_name)
ax.set_title('성별에 따른 입양 비율')
ax.set_xlabel('성별')


# +
# 입양동물 나이 분포
age_x = ['1', '2', '3', '4', '5', '6']
ax_age = df_pet.age.value_counts().iloc[:6].plot(kind = 'bar', color = 'orange')

ax_age.set_xticklabels(age_x, rotation=30)
ax_age.set_title('입양 동물 나이 분포')
ax_age.set_xlabel('나이')
ax_age.set_ylabel('마리')
plt.show()


# -

# ## 7. 모델 선정
# ###### 교차 검증으로 분류 모델 점수 비교 및 선정 

# +
# Decision Tree Classifier 
dt_clf = DecisionTreeClassifier(random_state=156)

#Logistic Regression - 개별 모델
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=156)

#K-최근접 이웃 분류기 모델 - 개별 모델
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=8)

# RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=156)

# 개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기 
from sklearn.ensemble import VotingClassifier 
vo_clf = VotingClassifier( estimators = [('LR', lr_clf), ('KNN',knn_clf)], voting='soft')

# Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(random_state=156)

# XGBOOST
import xgboost as xgb
from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=156)

# LIGHT GBM 
from lightgbm import LGBMClassifier
lgbm_wrapper = LGBMClassifier(n_estimators=400)

# GaussianNB
from sklearn.naive_bayes import GaussianNB
clf_pf = GaussianNB()

# SVC
from sklearn.svm import SVC
clf_svc = SVC(random_state=156)

# category boosting
# cat_bo = CatBoostClassifier(random_state=156)
# -

model_list = [lgbm_wrapper, dt_clf, lr_clf, rf_clf, gb_clf, xgb_model, clf_pf, clf_svc, vo_clf, knn_clf]

# +

for model in model_list:
    print(f'{model}_____________________________')
    result = cross_validate(model, features, label_adopt, 
               scoring={'정확도': 'accuracy',
                        'ROC-AUC':'roc_auc',
                        'f1': 'f1',
                        '정밀도':'precision',
                        '재현율(민감도)':'recall', 
                        'neg_brier_score':'neg_brier_score',
                        'neg_log_loss':'neg_log_loss',
                        'jaccard':'jaccard',
#                         'confusion' : 'confusion_matrix'
                       }, 
               return_train_score=False, 
               cv=5)
    print(result)

# +
# lgbm_wrapper + adopt

cross_validate(lgbm_wrapper, features, label_adopt, 
               scoring={'정확도': 'accuracy',
                        'ROC-AUC':'roc_auc',
                        'f1': 'f1',
                        '정밀도':'precision',
                        '재현율(민감도)':'recall', 
#                         'neg_brier_score':'neg_brier_score',
                        'neg_log_loss':'neg_log_loss',
#                         'jaccard':'jaccard',
#                         'confusion' : 'confusion_matrix'
                       }, 
               return_train_score=False, 
               cv=3)


# +
# from CatBoost import CatBoostClassifier
# Catboostclassifier 점수내기 

cross_validate(cat_bo, features, label_adopt, 
               scoring={'정확도': 'accuracy',
                        'ROC-AUC':'roc_auc',
                        'f1': 'f1',
                        '정밀도':'precision',
                        '재현율(민감도)':'recall', 
#                         'neg_brier_score':'neg_brier_score',
                        'neg_log_loss':'neg_log_loss',
#                         'jaccard':'jaccard',
#                         'confusion' : 'confusion_matrix'
                       }, 
               return_train_score=False, 
               cv=10)

# +
#최종 점수 비교 

# +
LGBMClassifier_result = {'fit_time': ([10.17200637,  9.4461    ,  9.33937812,  9.44675612,  9.22282577]),
 'score_time': ([0.70611119, 0.68616557, 0.67237639, 0.69515228, 0.67320013]),
 'test_정확도': ([0.76603853, 0.76584155, 0.76397444, 0.76491762, 0.76597629]),
 'test_ROC-AUC': ([0.79184431, 0.79233852, 0.79057673, 0.7876541 , 0.79167294]),
 'test_f1': ([0.49457358, 0.49587667, 0.49284473, 0.49101896, 0.49324775]),
 'test_정밀도': ([0.67342317, 0.67126669, 0.66495536, 0.67118605, 0.67430199]),
 'test_재현율(민감도)': ([0.39078723, 0.39315285, 0.39151005, 0.38710737, 0.38884143]),
 'test_neg_brier_score': ([-0.16002624, -0.16019638, -0.16069881, -0.16117585, -0.16033621]),
 'test_neg_log_loss': ([-0.48681337, -0.48688473, -0.48867188, -0.4902294 , -0.48746971]),
 'test_jaccard': ([0.32852723, 0.3296782 , 0.32700329, 0.3253977 , 0.32735823])}

DecisionTreeClassifier_result = {'fit_time': ([4.02026987, 3.71726608, 3.79908514, 3.70016623, 3.76344824]),
  'score_time': ([0.15951395, 0.15452695, 0.16456747, 0.16256928, 0.16357017]),
  'test_정확도': ([0.59554057, 0.59355415, 0.50582076, 0.59017555, 0.54538035]), 
  'test_ROC-AUC': ([0.48703175, 0.47717041, 0.51624564, 0.54517425, 0.54870819]), 
  'test_f1': ([0.24587999, 0.21975228, 0.39018736, 0.38399222, 0.41581416]), 
  'test_정밀도': ([0.27089264, 0.25104687, 0.3055291 , 0.34302374, 0.3333968 ]),
  'test_재현율(민감도)': ([0.22509594, 0.19539505, 0.53974345, 0.43607402, 0.55236042]),
  'test_neg_brier_score': ([-0.40418183, -0.40626151, -0.49337508, -0.40929703, -0.45305136]),
  'test_neg_log_loss': ([-13.94344481, -14.02308476, -17.00033797, -14.09242677, -15.57362297]), 
  'test_jaccard': ([0.14017284, 0.12343916, 0.2423806 , 0.23761781, 0.26247814])}

LogisticRegression_result = {'fit_time': ([2.42261505, 2.36415195, 2.58508563, 2.28887844, 5.27289844]),
  'score_time': ([0.14765978, 0.14871049, 0.15169978, 0.14760494, 0.1505971 ]),
  'test_정확도': ([0.71059885, 0.71358618, 0.68979058, 0.71088697, 0.7197567 ]), 
  'test_ROC-AUC': ([0.66066603, 0.6442938 , 0.65647586, 0.6402152 , 0.67901843]), 
  'test_f1': ([0.10085159, 0.17640808, 0.35281267, 0.2941464 , 0.18040982]), 
  'test_정밀도': ([0.56093667, 0.55939343, 0.45361421, 0.51629933, 0.62928055]), 
  'test_재현율(민감도)': ([0.05540661, 0.10471534, 0.28866576, 0.20565661, 0.10529913]), 
  'test_neg_brier_score': ([-0.19682781, -0.19662177, -0.20035454, -0.19960378, -0.19018093]), 
  'test_neg_log_loss': ([-0.58485709, -0.58981382, -0.59215156, -0.58707172, -0.56535987]),
  'test_jaccard': ([0.05310359, 0.0967366 , 0.21419098, 0.17243355, 0.0991486 ])}


RandomForestClassifier_result = {'fit_time': ([61.18253326, 59.13120317, 59.15694451, 61.72533536, 73.52390623]), 
  'score_time': ([3.0020299 , 3.04039454, 3.10249853, 2.67878413, 3.77390671]),
  'test_정확도': ([0.70945936, 0.62530605, 0.58329227, 0.66832461, 0.60994764]), 
  'test_ROC-AUC': ([0.63558102, 0.56458481, 0.58038719, 0.60633243, 0.62629082]), 
  'test_f1': ([0.05014096, 0.11235545, 0.37354446, 0.37825823, 0.42914451]),
  'test_정밀도': ([0.5921522 , 0.18355185, 0.3337332 , 0.41943538, 0.37558185]), 
  'test_재현율(민감도)': ([0.02617884, 0.08095463, 0.42414047, 0.34444328, 0.50052571]), 
  'test_neg_brier_score': ([-0.2046442 , -0.24410966, -0.24721153, -0.21929385, -0.23480853]),
  'test_neg_log_loss': ([-0.64348016, -0.74943459, -0.69719288, -0.68341953, -0.6773134 ]), 
  'test_jaccard': ([0.02571517, 0.05952151, 0.2296678 , 0.23324196, 0.27319159])}

GradientBoostingClassifier_result ={'fit_time': ([119.02110672, 121.23281145, 120.31095576, 121.39992571,121.36007309]),
 'score_time': ([1.14094877, 1.13696003, 0.89959407, 1.16887403, 1.15062666]),
 'test_정확도': ([0.71038327, 0.6673596 , 0.642085  , 0.6710194 , 0.73016631]), 
 'test_ROC-AUC': ([0.68641852, 0.60011153, 0.62337993, 0.61942679, 0.71909277]), 
 'test_f1': ([0.05601285, 0.10402323, 0.3619994 , 0.32660909, 0.29430953]),
 'test_정밀도': ([0.61931188, 0.24651071, 0.3787696 , 0.40782431, 0.62902393]), 
 'test_재현율(민감도)': ([0.02933291, 0.0659202 , 0.34665125, 0.27236884, 0.19209337]), 
 'test_neg_brier_score': ([-0.19798692, -0.20993889, -0.2180108 , -0.21428557, -0.18518386]),
 'test_neg_log_loss': ([-0.58281981, -0.60781563, -0.62375612, -0.61952705, -0.55263983]), 
 'test_jaccard': ([0.02881338, 0.05486524, 0.22100077, 0.195178  , 0.17254569])}

XGBClassifier_result = {'fit_time': ([92.53778005, 87.48144674, 87.95690417, 86.75425839, 88.42293835]), 
                        'score_time': ([1.03591156, 0.74500823, 0.68018079, 0.88463426, 0.67120481]), 
                        'test_정확도': ([0.71143038, 0.6695462 , 0.64644287, 0.66815522, 0.73270711]),
                        'test_ROC-AUC': ([0.69167708, 0.60683265, 0.62485801, 0.63211538, 0.72656394]),
                        'test_f1': ([0.05772325, 0.10835965, 0.35718685, 0.37752744, 0.3153743 ]),
                        'test_정밀도': ([0.66358382, 0.25847374, 0.38206756, 0.41896397, 0.63139608]),
                        'test_재현율(민감도)': ([0.030174  , 0.0685486 , 0.33534854, 0.34354957, 0.21017769]), 
                        'test_neg_brier_score': ([-0.19967648, -0.20789314, -0.21640833, -0.21307151, -0.18385024]),
                        'test_neg_log_loss': ([-0.58957457, -0.60284024, -0.6202125 , -0.61687932, -0.54959954]), 
                        'test_jaccard': ([0.02971937, 0.05728343, 0.21742391, 0.23268649, 0.18720734])}


GaussianNB_result = {'fit_time': ([0.13763165, 0.13663316, 0.1336422 , 0.13563704, 0.13663411]), 
  'score_time': ([0.1486032 , 0.12267208, 0.13663459, 0.12267303, 0.12267184]),
  'test_정확도': ([0.70579449, 0.70525554, 0.4526024 , 0.70602094, 0.70551278]), 
  'test_ROC-AUC': ([0.6350354 , 0.57387193, 0.61858796, 0.55316779, 0.63230338]), 
  'test_f1': ([0.00468848, 0.00674589, 0.46711039, 0.01088027, 0.0085027 ]),
  'test_정밀도': ([0.26011561, 0.26209677, 0.32672063, 0.37634409, 0.30827068]),
  'test_재현율(민감도)': ([0.00236556, 0.00341692, 0.81905162, 0.00551992, 0.0043108 ]),
  'test_neg_brier_score': ([-0.29394096, -0.29447555, -0.37548872, -0.29251871, -0.2943977 ]),
  'test_neg_log_loss': ([-2.29805515, -2.31803197, -1.11098036, -2.21793639, -2.64616697]),
  'test_jaccard': ([0.00234975, 0.00338436, 0.3047254 , 0.00546989, 0.0042695 ])}




KNN_result =  { 'test_ROC-AUC': ([0.57822838, 0.58108083, 0.49805141]),
 'test_f1': ([0.20186432, 0.27672481, 0.26921818]),}

Category_boost = {'test_ROC-AUC': ([0.76855011, 0.35962344, 0.47572046, 0.28703287, 0.3318827 ,
        0.5481514 , 0.5847507 , 0.52042764, 0.59795403, 0.73548871]),
 'test_f1': ([0.00753927, 0.22428541, 0.14208314, 0.09637566, 0.16354202,
        0.35422243, 0.386894  , 0.2491996 , 0.30040538, 0.54547127])}

voting = {'fit_time': ([378.3397069 , 387.08721757, 399.05286145, 379.98279619, 392.11164904]),
        'score_time': ([1316.45910239, 1553.85220003, 2078.49921322, 2792.74337053,5500.73604035]),
        'test_정확도': ([0.70539413, 0.6860227 , 0.65344934, 0.69180782, 0.69685864]), 
        'test_ROC-AUC': ([0.60461103, 0.59880446, 0.62701032, 0.63588207, 0.66537769]),
         'test_f1': ([0.11972025, 0.1834855 , 0.39370673, 0.31781308, 0.36709105]), 
          'test_정밀도': ([0.47989672, 0.38510674, 0.40376858, 0.45191935, 0.47252111]), 
          'test_재현율(민감도)': ([0.0683909 , 0.12043316, 0.38413416, 0.24508464, 0.30012617]),
          'test_neg_brier_score': ([-0.20865921, -0.21143363, -0.21681568, -0.20172261, -0.19723774]),
          'test_neg_log_loss': ([-0.61927783, -0.62163431, -0.62383438, -0.59151999, -0.58009615]), 
          'test_jaccard': ([0.06367151, 0.10100966, 0.24510264, 0.18892851, 0.22480803])}
# -

# 점수 확인 --> LGBM 이 가장 높게 나온 것을 확인 
pd.Series(LGBMClassifier_result.get('test_f1')).mean(), pd.Series(DecisionTreeClassifier_result.get('test_f1')).mean(), pd.Series(LogisticRegression_result.get('test_f1')).mean(),pd.Series(RandomForestClassifier_result.get('test_f1')).mean(),pd.Series(GradientBoostingClassifier_result.get('test_f1')).mean(),pd.Series(XGBClassifier_result.get('test_f1')).mean(),pd.Series(GaussianNB_result.get('test_f1')).mean(),pd.Series(KNN_result.get('test_f1')).mean(),pd.Series(Category_boost.get('test_f1')).mean(),pd.Series(voting.get('test_f1')).mean()

model = [LGBMClassifier_result,
        DecisionTreeClassifier_result,
        LogisticRegression_result,
        RandomForestClassifier_result,
        GradientBoostingClassifier_result,
        XGBClassifier_result,
        GaussianNB_result,
        KNN_result,
        Category_boost,
        voting]

# +
roc_score = []
f1_score = []

for i in model:
    
    roc_mean = pd.Series(i['test_ROC-AUC']).mean()
    f1_mean = pd.Series(i['test_f1']).mean()
    roc_score.append(roc_mean)
    f1_score.append(f1_mean)
    
print('roc_score:', roc_score)
print('f1_score:', f1_score)
    
    
# -

model_name = ['LGBMClassifier_result',
        'DecisionTreeClassifier_result',
        'LogisticRegression_result',
        'RandomForestClassifier_result',
        'GradientBoostingClassifier_result',
        'XGBClassifier_result',
        'GaussianNB_result',
        'KNN_result',
        'xgboost_result',
        'category_boost']

data = {'roc_score':roc_score, 'f1_score': f1_score, 'model_name':model_name}
final_data = pd.DataFrame.from_dict(data)
final_data

# +
# sns.set_theme(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Plot the total crashes
sns.set_color_codes("pastel")

sns.barplot(x="roc_score", y="model_name", data=final_data,
            label="roc_score", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="f1_score", y="model_name", data=final_data,
            label="f1_score", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="model_name",
       xlabel="roc_score,f1_score")
sns.despine(left=True, bottom=True)
# -

# ## 7. 하이퍼 파리미터 조정 

# +
# 1) 기본 파라미터  --> F1: 0.44145508328344546
lgbm_wrapper = LGBMClassifier(random_state=156)


# 2) 파라미터 조정 후  --> F1: 0.5588771418155305
# lgbm_wrapper = LGBMClassifier(n_estimators=50429, learning_rate=1, max_depth=100, num_leaves= 5000, random_state=156)

evals = [(X_valid, y_valid)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='f1_score', eval_set=evals, verbose=True)

preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:,1]


# -

# 평가 점수 함수 생성
def get_clf_eval(y, pred, proba=None) :
    print(confusion_matrix(y, pred))
    print("정확도:", accuracy_score(y, pred))
    print("F1:", f1_score(y, pred))
    print("정밀도:", precision_score(y, pred))
    print("재현률(민감도):", recall_score(y, pred))
    print("ROC:", roc_auc_score(y, pred))


# 1) 기본 파라미터 : 예측 성능 평가  
get_clf_eval(y_test, preds, pred_proba)

# +
# GridSearchCV로 하이퍼파라미터 조정 

# +
lgbm_wrapper = LGBMClassifier(n_estimators=50429, random_state=156)

params = { 'num_leaves': [7000, 8000, 9000],
         'max_depth': [70, 80, 90],
          'learning_rate': [0.1, 0.15, 0.2]
#          'min_child_samples':[60,100],
#          'subsample':[0.8,1]
         }

gridcv = GridSearchCV(lgbm_wrapper, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='f1_score',
          eval_set=[(X_train, y_train), (X_valid, y_valid)])

# print('GridSearchCV 최적의 파라미터: ', gridcv.best_params_)
# score = f1_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')
# score

# +
# 최적의 파리미터 도출 
gridcv.best_params_ 

# 1차 시도 {'max_depth': 100, 'num_leaves': 10000}
# 2차시도 {'learning_rate': 0.2, 'max_depth': 80, 'num_leaves': 8000}
# 3차 시도 {'learning_rate': 0.1, 'max_depth': 70, 'num_leaves': 7000}
# ...........
# -

# 2) 파라미터 조정 후 : 예측 성능 평가  
get_clf_eval(y_test, preds, pred_proba)

# +
# 중요 피쳐 시각화 
from lightgbm import plot_importance
import matplotlib.pyplot as plt
# %matplotlib inline

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(lgbm_wrapper, ax=ax)


# -

# ## 8. 임곗값 선정 및 최종 점수 도출 

# +
# f1 score 임곗값 조정 

def precision_recall_curve_plot(y_test, pred_proba_c1):
    #threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    #X축을 threshold값으로, y축은 정밀도, 재현율 값으로 각각 plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle = '--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label = 'recall')
    
    # threshold값 x축의 scale을 0.1단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    
    #x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot(y_test, pred_proba)

# +
threshold = [0.2,0.29, 0.295, 0.3,0.31,0.35, 0.36, 0.37]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값: ', custom_threshold)
        get_clf_eval(y_test, custom_predict)


# -

# threshold 조정 : 최종 점수 --> F1: 0.6133493131741495
get_eval_by_threshold(y_test, pred_proba.reshape(-1,1), threshold) #0.295




