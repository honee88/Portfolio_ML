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

# # 2. EDA: 데이터 전처리  
# #### 2020_10_21

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import re
import ast # 형변환 

from pprint import pprint as pp
import warnings
warnings.filterwarnings(action='ignore')

# 텍스트 마이닝
from eunjeon import Mecab
import nltk
from konlpy.tag import Okt
from nltk import FreqDist
from gensim.models import Word2Vec

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

# ## 1. 데이터 불러오기 

# +
# import data
df_raw = pd.read_csv("./combined_final_20170101_20201013")

# 활용 데이터 프레임화 
df = df_raw
del df['Unnamed: 0']
df.info()
# -

df.shape

df.head()

# ## 2. 변수 탐색 

# ## (변수 측정 방법 및 척도 수준)
# ### 1. 질적 변수 
# ##### desrtionno: 유기번호 -  명목척도 / 독립변수
# ##### happenPlace	발견장소 - 명목척도  / 독립변수
# ##### kindCd	품종 - 명목척도  / 독립변수
# ##### colorCd	색상 - 명목척도  / 독립변수
# ##### processState 상태  - 명목척도 / 종속변수
# ##### sexCd	성별  - 명목척도  / 독립변수 
# ##### neuterYn 중성화여부  - 명목척도  / 독립변수
# ##### specialMark	특징  - 명목척도  / 독립변수
# ##### careNm	보호소이름  - 명목척도  / 독립변수
# ##### orgNm	관할기관 - 명목척도  / 독립변수			
# ##### type 동물종류 - 명목척도 / 독립변수 
# ##### noticeNo 공고번호 - 명목척도 / 독립변수																				
#
# ### 2. 양적 변수 
# ##### age	나이 - 간격척도  / 독립변수
# ##### happenDt: 접수일 - 분석 목적에 따라 순서/간격 척도  / 독립변수
# ##### noticeSdt	공고시작일 - 분석 목적에 따라 순서/간격 척도  / 독립변수
# ##### noticeEdt	공고종료일 - 분석 목적에 따라 순서/간격 척도  / 독립변수

# ## 3. 결측치/이(상한)상치 처리

# +
# 결측치 확인  
df.noticeno.isnull().unique()
df[df['noticeno'].isnull() == True]
# 결측치 처리 
# df.noticeno = df.noticeno.fillna('9999')
df.noticeno = df.noticeno.fillna(np.nan)

# # 삭제 
# a = df[df['noticeno'].isnull() == True].index
# df = df.drop(a)

# +
# 날짜 데이터 이상치 확인 및 데이터 변경 

df.iloc[231419, :] #2170918 --> '2017098' 
df.noticeedt[231419] = '20170918'

df.iloc[231489, :] #2170921 --> '20170921' 
df.noticeedt[231489] = '20170921' 


# +
# 년생 age 이상치 확인 및 데이터 삭제 처리 

def age_in_year(born_year):
    p = re.compile('\d{4}\(\년\생\)')
    m = p.match(born_year)
    if m:
        return(born_year)
    else:
        return('0')

q = df.age
w = list(q.map(age_in_year))
e = w.count('0')

# e = list(df.age.map(age_in_year)).count('0')

z = []
for i in range(0, len(w)):
    if w[i] == '0':
        z.append(i)

# df = df.drop(z) # 행 삭제

df['age'] = df['age'].map(lambda x: x[0:4])
# -

# 년생 결측치 처리 
df.age[z] = np.nan

df['age'].unique()


# ##  4. 데이터 형변환
# ##### 데이터 변수 성격에 따른 알맞은 데이터 형변환

# +
# 날짜 타입으로 바꾸기 

def int_to_date(col_name):
    col_name = col_name.astype(str)
    col_name = col_name.str[0:4] + "/" + col_name.str[4:6] + "/" + col_name.str[6:8] 
    col_name = col_name.astype('datetime64[ns]')
    
    return col_name

df.happendt = int_to_date(df.happendt)
df.noticesdt = int_to_date(df.noticesdt)
df.noticeedt = int_to_date(df.noticeedt)

## 참고: 날짜 데이터 년/월/일 추출 함수 
# df.happendt.dt.year
# df.happendt.dt.month
# df.happendt.dt.day

# -

# ##  5. 질적 변수 : 데이터 범주화 categorizing/factoring

df.processstate.value_counts()


# +
# 상태 범주화 : 입양 1, 나머지 0
def processstate_to_int(processstate):
    
    if processstate =='종료(입양)':
        return 1
    else: 
        return 0
    
df['processstate_adopt'] = df.processstate.map(processstate_to_int)
df.processstate_adopt = df.processstate_adopt.fillna('0')
df.processstate_adopt.value_counts()


# +
# 상태 범주화 : 입양 1, 나머지 0
def processstate_to_int(processstate):
    
    if processstate =='종료(안락사)':
        return 1
    else: 
        return 0
    
df['processstate_euthanize'] = df.processstate.map(processstate_to_int)
df.processstate_euthanize = df.processstate_euthanize.fillna('0')
df.processstate_euthanize.value_counts()


# +
# 성별 범주화 
def gender_to_int(gender):

    if gender=='F':
        gender = 1
    elif gender=='M':
        gender = 0
    else:
        gender = 2
    return gender

df['sexcd'] = df.sexcd.map(gender_to_int)


# +
# 중성화 여부
def neuteryn_to_int(neuteryn):

    if neuteryn=='N':
        neuteryn = 0
    elif neuteryn =='Y':
        neuteryn = 1
    else:
        neuteryn = 2
    return neuteryn

df['neuteryn'] = df.neuteryn.map(neuteryn_to_int)
# -

# 반려동물 타입 : 1 = dog, 0 = cat
df.type = df.type.apply(lambda x : '1' if x == 'dog' else '0')

# #### ( 색상 color : 범주형 데이터 수치화 )

# +
# 상위 5개 색상으로 구분: 검정 / 흰색 / 갈색 / 회색 / 노랑 
# 5개 색상 컬럼 생성 후, 해당되는 색 컬럼에 1 표시
# 즉, 여러 색상 포함하는 경우 해당 모든 컬럼에 1 표시


# 결측치: 5개 컬럼 모두 0으로 표시
# 1) 상위 5개 색상에 포함되지 않는 경우: ex) 살구색, 블루, 분홍색...
# 2) 색상 판단 불가한 경우: ex) 점박이, 삼색, 파티색, 얼룩무늬...
# 3) 색상이 아닌 주소/성격/품종/나이 등이 입력된 경우
# 4) 숫자 또는 공백이 입력된 경우


# 강아지+고양이 총 324,657행 중 16,087행이 결측치

# 색 분류 참고
# 고등어 -> brown
# 호구   -> brown, yellow
# 호피   -> brown, black
# 재구   -> brown, yellow
# -

# 색깔 파생변수 생성 
df['blc_col'] = 100
df['brown_col'] = 100
df['white_col'] = 100
df['grey_col'] = 100
df['yellow_col'] = 100


# +
# 특수문자 제거 
def cleanse(x):
    x = x.strip()
    x = x.replace(".", "")
    x = x.replace("/", "")
    x = x.replace("+", "")
    x = x.replace(",", "")
    x = x.replace("&", "")
    x = x.replace("-", "")
    x = x.replace(" ", "")
    x = x.replace("(", "")
    x = x.replace(")", "")
    x = x.replace(";", "")
    return x

df['colorcd'] = df.colorcd.map(cleanse)
# -

a = df.colorcd.unique()
clist = a.tolist()
clist


def blc_tf(x):

        # ---------------------------------------- 검정 
        # 조심: 흑갈
        if x.find('검') + x.find('흑') + x.find('블랙') + x.find('black') + x.find('Black') + x.find('BLACK') + x.find('건정') + x.find('겁정') + x.find('까') + x.find('껌') + x.find('달마시안') + x.find('바둑') + x.find('불랙') + x.find('검졍') + x.find('젖소') + x.find('ㄱ머정') + x.find('감정') + x.find('거멍') + x.find('거멎ㅇ') + x.find('거정') + x.find('걸정') + x.find('블렉') + x.find('젓소') + x.find('깜') + x.find('블랜') + x.find('호피') + x.find('컴정') + x.find('훅색') > -28:
            return 1
        else: 
            return 0


def white_tf(x):

        # ---------------------------------------- 흰색 
        if x.find('흰') + x.find('백') + x.find('하') + x.find('벡') + x.find('크림') + x.find('아이보리') + x.find('희') + x.find('흐니') + x.find('휜') + x.find('white') + x.find('White') + x.find('WHITE') + x.find('힌') + x.find('크김') + x.find('화이트') + x.find('흐ㅟ') + x.find('달마시안') + x.find('바둑') + x.find('하얀') + x.find('환색') + x.find('횐') + x.find('흔색') + x.find('흼') + x.find('읜색') + x.find('읜') + x.find('젖소') + x.find('크리') + x.find('ㅎㄴ;섹') + x.find('하얀색') + x.find('흐ㅏㄴ') + x.find('희') + x.find('Ivory') + x.find('ivory') + x.find('젓소') + x.find('ㅎㄴ섹') + x.find('하얀색') + x.find('상아') + x.find('이아보리') + x.find('햐얀') + x.find('후ㅏㄴ샥') + x.find('힁') > -41:
            return 1
        else: 
            return 0


def grey_tf(x):

        # ---------------------------------------- 회색 
        if x.find('쥐') + x.find('회') + x.find('은색') + x.find('은회') + x.find('실버') + x.find('재색') + x.find('그레이') + x.find('잿') + x.find('제색') + x.find('차콜') + x.find('grey') + x.find('gray') + x.find('먹색') + x.find('홰색') + x.find('화색') + x.find('화섹') > -16 and x.find('검은색') == -1:
            return 1
        else: 
            return 0
        if x.find[0] == '은':
            return 1


def brown_tf(x):

        # ---------------------------------------- 갈색
        if x.find('갈') + x.find('밤') + x.find('브라운') + x.find('초코') + x.find('고동') + x.find('흑갈') + x.find('베이지') + x.find('갈흑') + x.find('연갈') + x.find('쵸코') + x.find('초콜') + x.find('황토') + x.find('배이지') + x.find('길색') + x.find('apricot') + x.find('Apricot') + x.find('rkftor') + x.find('SaltPepper') + x.find('강객') + x.find('강색') + x.find('걀색') + x.find('걸샥') + x.find('길섹') + x.find('에프리코트') + x.find('에프리푸들') + x.find('연베이') + x.find('커피') + x.find('흙') + x.find('Tan') + x.find('tan') + x.find('brown') + x.find('Brown') + x.find('BROWN') + x.find('초쿄') + x.find('칡') + x.find('코코아') + x.find('가랙') + x.find('간색') + x.find('감색') + x.find('애프리') + x.find('애프리콧') + x.find('에프리') + x.find('고등어') + x.find('고긍어') + x.find('고둥어') + x.find('고드엉') + x.find('고등') + x.find('고등러') + x.find('고등색') + x.find('고릉어') + x.find('공등어') + x.find('탄') + x.find('호피') + x.find('호구') + x.find('재구') + x.find('카카오') > -56:
            return 1
        else: 
            return 0


def yellow_tf(x):

        # ---------------------------------------- 노랑
        # 조심: 황토
        if x.find('황') + x.find('노랑') + x.find('골든') + x.find('옐') + x.find('금') + x.find('골') + x.find('누렁') + x.find('노란') + x.find('누') + x.find('치즈') + x.find('GOLD') + x.find('노') + x.find('모래') + x.find('yellow') + x.find('Yellow') + x.find('YELLOW') + x.find('ㅊㅣ즈') + x.find('치츠') + x.find('치느') + x.find('츠지') + x.find('gold') + x.find('ㄴ랑') + x.find('호구') + x.find('재구') > -24:
            return 1
        else: 
            return 0


df['blc_col'] = df.colorcd.map(blc_tf)

df['white_col'] = df.colorcd.map(white_tf)

df['grey_col'] = df.colorcd.map(grey_tf)

df['brown_col'] = df.colorcd.map(brown_tf)

df['yellow_col'] = df.colorcd.map(yellow_tf)

df.info()

# #### ( 품종 데이터 knm/kindcd : 범주형 데이터 수치화 )

# +
# 원본 데이터 <-> 품종코드 데이터 컬럼명 동일하게 수정
df = df.rename({'kindcd':'knm'}, axis = 'columns')

# dog/cat 분류 
dog = df.iloc[:212618]
cat = df.iloc[212618:]

# 개/ 고양이 품종 데이터 import 
dog_kind = pd.read_csv('./dog_kind')
cat_kind = pd.read_csv('./cat_kind')

dog_kind.drop('Unnamed: 0', axis = 1, inplace = True)
cat_kind.drop('Unnamed: 0', axis = 1, inplace = True)
# -

# 전처리 : 개/고양이 각 테이블의 [개]/[고양이] 문자열 strip (변수에 담아서 확인)
dog['knm'] = dog['knm'].str.lstrip('[개] ')
cat['knm'] = cat['knm'].str.lstrip('[고양이] ')

#  중복값 처리 : 도고 아르젠티노 kindcd 153번 삭제
dog_kind.drop(15, axis = 0, inplace = True )

# +
# [고양이]로만 기재된 데이터는 믹스로 추정하여 코드부여
cat_kind.loc[36] = ['', 212]

# '페르시안페르시안 친칠라' 코드 페르시안페르시안-친칠라와 동일하게 부여
cat_kind.loc[37] = ['페르시안페르시안 친칠라', 197]
# -

## 공백, 특수 문자 제거
dog['knm'].str.strip()
cat['knm'].str.strip()


# +
def cleanText(dataframe):
 
    #텍스트에 포함되어 있는 특수 문자 제거
 
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', dataframe)
 
    return text

dog['knm'] = dog.knm.map(cleanText)
cat['knm'] = cat.knm.map(cleanText)
# -

# knm을 key로 merge 진행
dog = pd.merge(dog, dog_kind, how='left', left_on='knm', right_on='knm')
cat = pd.merge(cat, cat_kind, how='left', left_on='knm', right_on='knm')

dog.kindcd.value_counts()

# merge후 kindcd null값 확인
dog.kindcd.isnull().sum() # 2595

# merge후 kindcd null값 확인
cat.kindcd.isnull().sum() # 4299

# null 값인 행 눈으로 확인
dog[dog['kindcd'].isnull() == True]


# +
# 강아지
# -

# null 대부분이 '믹스'관련인 것 파악 
# 믹스관련 강아지에 품종 코드 부여: mapping 방식 적용할 수 있는 함수 dog_mix --> 훨씬 빠름!
def dog_mix(x):
    
    if (x.find('믹스견') + x.find('mix') + x.find('잡종') +
       x.find('믹스') + x.find('혼종') + x.find('혼합')+
       x.find('+') + x.find('추정') + x.find('?')+
       x.find('발바리')  + x.find('기타') + x.find('믹')+ x.find('49856') + x.find('잡견')) > -14:
        return '믹스견'

    else:
        return x


def huski_dog(x):
    
    if x.find('허스') > -1:
        return '시베리안 허스키'

    else:
        return x


# 함수 mapping
dog['knm'] = dog.knm.map(dog_mix)
dog['knm'] = dog.knm.map(huski_dog)

dog.info()


# +
# 고양이
# -

# 믹스관련 고양이에 품종 코드 부여: mapping 방식 적용할 수 있는 함수 cat_mix --> 훨씬 빠름!
def cat_mix(x):
    
    if (x.find('믹스묘') + x.find('mix') + x.find('잡종') +
       x.find('믹스') + x.find('혼종') + x.find('혼합')+
       x.find('+') + x.find('추정') + x.find('?')+
       x.find('발바리')  + x.find('기타') + x.find('믹') + x.find('54638')) > -13:
        return '믹스묘'

    else:
        return x


# 품종 '한국 고양이' 전처리
def korea_cat(x):
    
    if (x.find('코숏') + x.find('한국')) > -2:
        return '한국 고양이'

    else:
        return x


# 함수 mapping 
cat['knm'] = cat.knm.map(cat_mix)
cat['knm'] = cat.knm.map(korea_cat)

# 품종 전처리 후 다시 merge - 코드 부여
dog = pd.merge(dog, dog_kind, how='left', left_on='knm', right_on='knm')
cat = pd.merge(cat, cat_kind, how='left', left_on='knm', right_on='knm')

dog.head()

# merge하면서 생긴 필요없는 컬럼 drop
dog.drop('kindcd_x', axis = 1, inplace = True)
cat.drop('kindcd_x', axis = 1, inplace = True)

# 컬럼명 재지정
dog = dog.rename({'kindcd_y':'kindcd'}, axis = 'columns')
cat = cat.rename({'kindcd_y':'kindcd'}, axis = 'columns')

# +
# null kindcd는 9999로 처리
dog.kindcd = dog.kindcd.fillna(np.nan)
dog.kindcd.astype('float')

cat.kindcd = cat.kindcd.fillna(np.nan)
cat.kindcd.astype('float')
# -

# 최종 dog, cat 데이터프레임 concat 진행
dog_cat = pd.concat([dog, cat])

df = dog_cat

# ##  6. 양적 변수 :  데이터 수정 

df.age.isna() == True

# +
# 나이 계산하기 - 태어난 년도를 n살로 바꾸기

null_list = df[df.age.isna() == True].index #결측 처리 데이터 9999 개수, 58개
len(null_list)
# -

# 태어난 연도 - 공지 기준 연도 + 1
df['age'] = pd.DatetimeIndex(df.noticesdt).year - df.age.map(float) +1

df.age = df.age.map(float)

df.age[df.age < 0 ] = np.nan

df.age.iloc[null_list] = np.nan

# +
# 체중 : 체중 표기 방법이 아닌경우, nan 리턴 


def weight_reg_01(x):
    
    x = str(x)
    idx = x.find('(') 
    x = x[:idx] 
    
    m = re.match('\d+\,+|\d+\;+|\d+\:+|\d+\/+|\d+\-+|\d+\~+', x) # 숫자 + 특수문자 

    if m:                                          # 특수 문자있는 경우 뒤에 절삭 
        start, end = m.span()
        x = x[:start+1]
#         x = x.replace(x[start+1:end],'') 
        return x
    
    else:
        return x

    
def weight_reg_02(x):
    
    if x.__contains__("미"):  # 미확인 & 미상 처리 
        return 'NaN' 
    
    else:
        x = x.rstrip('.')
        return x 

df['weight'] = df.weight.map(weight_reg_01)
df['weight'] = df.weight.map(weight_reg_02)

# +
# weight_float 새로운 변수로 생성해둠 

a= []

for i in list(df.weight):
    try:
        a.append(float(i))
    except:
        a.append(np.nan)
        
a.count(np.nan)
df['weight_float'] = a
# -

df['noticesdt_year'] = df.noticesdt.dt.year
df['noticesdt_month'] = df.noticesdt.dt.month
df['noticesdt_day'] = df.noticesdt.dt.day

# ## 7. 텍스트 마이닝 : 유기동물 특징 분류 specialmark          

# ### 7-1)  specialmark feature 데이터 탐색

df['specialmark'].value_counts()

# ### 7-1)  specialmark feature EDA 
# ##### 텍스트 길이 분포 확인

# +
test_lenght = df['specialmark'].astype(str).apply(len)

plt.figure(figsize=(12,5))

plt.hist(test_lenght, bins=200, alpha=0.5, color='r', label='word')

plt.title('Text length Distribution')
plt.xlabel('Length of specialmark')
plt.ylabel('Number of specialmark')

plt.show()

# +
print('특징 길이 최대 값: {}'.format(np.max(test_lenght)))
print('특징 길이 최소 값: {}'.format(np.min(test_lenght)))
print('특징 길이 평균 값: {:.2f}'.format(np.mean(test_lenght)))
print('특징 길이 표준편차: {:.2f}'.format(np.std(test_lenght)))
print('특징 길이 중간 값: {}'.format(np.median(test_lenght)))

# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('특징 길이 제 1 사분위: {}'.format(np.percentile(test_lenght, 25)))
print('특징 길이 제 3 사분위: {}'.format(np.percentile(test_lenght, 75)))
# -

# ##### Word 단위 텍스트 길이 분포

# +
test_word_counts = df['specialmark'].astype(str).apply(lambda x:len(x.split(' ')))

plt.figure(figsize=(12, 5))
plt.hist(test_word_counts, bins=50, facecolor='r',label='test')
plt.title('Log-Histogram of word count in specialmark', fontsize=15)
# plt.yscale('log', nonposy='clip')
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Number of specialmark', fontsize=15)

plt.show()
# -

print('특징 단어 개수 최대 값: {}'.format(np.max(test_word_counts)))
print('특징 단어 개수 최소 값: {}'.format(np.min(test_word_counts)))
print('특징 단어 개수 평균 값: {:.2f}'.format(np.mean(test_word_counts)))
print('특징 단어 개수 표준편차: {:.2f}'.format(np.std(test_word_counts)))
print('특징 단어 개수 중간 값: {}'.format(np.median(test_word_counts)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('특징 단어 개수 제 1 사분위: {}'.format(np.percentile(test_word_counts, 25)))
print('특징 단어 개수 제 3 사분위: {}'.format(np.percentile(test_word_counts, 75)))

# ### 7-3)  자연어 처리 
# #### 1) 텍스트 전처리 

text_df = df[['desertionno','specialmark','processstate']]

# 정규표현식, 한글과 공백을 제외하고 모두 제거 
text_df['specialmark'] = text_df['specialmark'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힝]","")

# #### 2) 불용어 정의 

stopwords=['의','가','며','들','는','됨','좀','걍','과','를','을','으로',
           '에','와','한','하다','개체','으나','관리','번호','면','함','쪽','줄','신고자'
           '혼종','고양이','묘','발견','추정','생후','개월','남음','믹스','구조','음','고']

# #### 3) 토큰화 (tokenized) - 형태소 분석기 1. Mecab

# +
tokenizer = Mecab()

tokenized=[]
for sentence in text_df['specialmark']:
    temp = []
    temp = tokenizer.morphs(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized.append(temp)
    
print(tokenized[:30])
# -

new = pd.DataFrame(tokenized)
new

# #### 3) 토큰화 (tokenized) - 형태소 분석기 2. Okt

sample_data = text_df.copy()

tokenizer2 = Okt()

tokenized2 = []
for sentence in sample_data['specialmark']:
    temp = []
    temp = tokenizer2.pos(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized2.append(temp)

# ### 7-4)  단어 집합(Vocabulary) 생성
# ### 중복을 제거한 텍스트 내 총 단어의 집합(set)

# +
# 단어-빈도수 조합으로 이루어진 단어집합 해시맵 vocab 생성

vocab = FreqDist(np.hstack(tokenized))
print('단어 집합의 크기 : {}'.format(len(vocab)))
# -

print('단어 [순]의 빈도수는? ', vocab['순'], '번')
print('단어 [온순]의 빈도수는? ', vocab['온순'], '번')
print('단어 [순하]의 빈도수는? ', vocab['순하'], '번')
print('단어 [순함]의 빈도수는? ', vocab['순함'], '번')

print('단어 [사나움]의 빈도수는? ', vocab['사나움'], '번')
print('단어 [사납]의 빈도수는? ', vocab['사납'], '번')
print('단어 [사나]의 빈도수는? ', vocab['사나'], '번')
print('단어 [사나운]의 빈도수는? ', vocab['사나운'], '번')

# ### 7-5)  워드 임배딩
# ### Word2Vec

model = Word2Vec(tokenized, size=100, window=3, min_count=100, workers=4, iter=100, sg=1)
# size = 100 : 컨텐츠를 100차원의 벡터로 변환
# window = 5 : 주변 단어 앞 뒤로 3개 까지 확인
# min_count = 100 코퍼스 내 출현 빈도가 100 번 미만인 단어는 분석에서 제외
# workers = 4 : CPU 쿼드코어로 사용
# iter = 100 : 100번 반복 학습
# sg = 1 : CBOW -> 0 Skip-Gram -> 1 중 Skip-Gram 선택 

model_result1=model.wv.most_similar("온순", topn=100)
pp(model_result1)

model_result3=model.wv.most_similar("귀여움", topn=100)
pp(model_result3)

# good_behavior 단어 도출
('순'),('순하'),('순한'),('온순'),('유순'),('조용'),('활달'),('활발'),('발랄'),('좋아하'),('따르'),('좋'),('따름'),('영리'),
('좋아함'),('친화'),('온'),('애교'),('착함'),('깨끗'),('착하'),('친숙'),('얌전'),('이쁨'),('귀여움'),('예쁨'),('귀여운'),
('귀여워'),('예쁘'),('매력'),('사랑'),('이쁘'),('이쁜'),('착한'),('호기심'),('밝'),('순둥이'),('아함')

model_result2=model.wv.most_similar("사나움", topn=100)
pp(model_result2)

model_result6=model.wv.most_similar("겁", topn=100)
pp(model_result6)

# sensitive 단어 도출
('사나움'),('사나운'),('예민'),('사납'),('경계'),('입질'),('공격'),('경계심'),('까칠'),('포악'),('위협'),('으르렁댐'),
('도망'),('소심'),('움찔'),('겁')

model_result4=model.wv.most_similar("사고", topn=100)
pp(model_result4)

# accident 단어 도출
('사고'),('교통사고'),('골절'),('골반'),('불능'),('기립'),('부상'),('예후'),('보행'),('마비'),('손상'),('출혈'),('혼수'),
('쓰러져'),('하반신'),('덫'),('올무'),('후구'),('후지'),('이상'),('당함'),('파열'),('다친'),('코마'),('척추'),('안면'),
('뒷다리'),('복수'),('의식'),('곤란'),('앞다리'),('신경'),('두부'),('척추골절'),('전지'),('응급'),('이송'),('아픈'),
('심각'),('소실'),('장기'),('병원'),('우측'),('괴사'),('물린'),('타박상'),('팽만'),('부종'),('탈장'),('돌출')

model_result5=model.wv.most_similar("질병", topn=100)
pp(model_result5)

# disease 단어 도출
('질병'),('전염성'),('질환'),('저하'),('감염'),('전염병'),('영양'),('노출'),('안질'),('원인'),('호흡기'),('탈수'),('증상'),
('저체온증'),('구내염'),('치석'),('진드기'),('모낭충'),('악액질'),('기생충'),('바이러스'),('영양실조'),('광견병'),('의심'),
('방치'),('눈병'),('기력'),('폐렴'),('장애'),('전신'),('움직임'),('외부'),('피부염'),('탈진'),('장염'),('쇠약'),('미상'),
('병심'),('곰팡이'),('비강'),('피부병'),('심장'),('사상충'),('쇄약'),('병심') 

# ### 7-6)  벡터화된 단어들로 Kmean Clustering

# +
word_vectors = model.wv.syn0

num_clusters = int(word_vectors.shape[0]/50)

print(num_clusters)
num_clusters = int(num_clusters)

# +
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

idx = list(idx)
names = model.wv.index2word
word_centroid_map = {names[i]: idx[i] for i in range(len(names))}
# -

for c in range(num_clusters):
    # 클러스터 번호 출력
    print("\ncluster {}".format(c))
    
    words = []
    cluster_values = list(word_centroid_map.values())
    for i in range(len(cluster_values)):
        if (cluster_values[i] == c):
            words.append(list(word_centroid_map.keys())[i])            
    print(words)

# ### 7-7)  PCA
#

# 특성 열 생성
df['good_behavior'] = 0
df['sensitive'] = 0
df['disease'] = 0
df['accident'] = 0

# +
good_behavior_list = [('순'),('순하'),('순한'),('온순'),('유순'),('조용'),('활달'),('활발'),('발랄'),('좋아하'),('따르'),('좋'),('따름'),('영리'),
('좋아함'),('친화'),('온'),('애교'),('착함'),('깨끗'),('착하'),('친숙'),('얌전'),('이쁨'),('귀여움'),('예쁨'),('귀여운'),
('귀여워'),('예쁘'),('매력'),('사랑'),('이쁘'),('이쁜'),('착한'),('호기심'),('밝'),('순둥이'),('아함')]

sensitive_list = [('사나움'),('사나운'),('예민'),('사납'),('경계'),('입질'),('공격'),('경계심'),('까칠'),('포악'),('위협'),('으르렁댐'),
('도망'),('소심'),('움찔'),('겁')]

disease_list = [('사고'),('교통사고'),('골절'),('골반'),('불능'),('기립'),('부상'),('예후'),('보행'),('마비'),('손상'),('출혈'),('혼수'),
('쓰러져'),('하반신'),('덫'),('올무'),('후구'),('후지'),('이상'),('당함'),('파열'),('다친'),('코마'),('척추'),('안면'),
('뒷다리'),('복수'),('의식'),('곤란'),('앞다리'),('신경'),('두부'),('척추골절'),('전지'),('응급'),('이송'),('아픈'),
('심각'),('소실'),('장기'),('병원'),('우측'),('괴사'),('물린'),('타박상'),('팽만'),('부종'),('탈장'),('돌출')]

accident_list = [('질병'),('전염성'),('질환'),('저하'),('감염'),('전염병'),('영양'),('노출'),('안질'),('원인'),('호흡기'),('탈수'),('증상'),
('저체온증'),('구내염'),('치석'),('진드기'),('모낭충'),('악액질'),('기생충'),('바이러스'),('영양실조'),('광견병'),('의심'),
('방치'),('눈병'),('기력'),('폐렴'),('장애'),('전신'),('움직임'),('외부'),('피부염'),('탈진'),('장염'),('쇠약'),('미상'),
('병심'),('곰팡이'),('비강'),('피부병'),('심장'),('사상충'),('쇄약'),('병심') ]

# +
for i in good_behavior_list:
    i.strip('()')
    
for i in sensitive_list:
    i.strip('()')  
    
for i in disease_list:
    i.strip('()')
    
for i in accident_list:
    i.strip('()')


# +
def good_behavior(x):
    
    for i in range(len(good_behavior_list)):
        if x.__contains__(good_behavior_list[i]):
            return 1

def sensitive(x):

    for i in range(len(sensitive_list)):
        if x.__contains__(sensitive_list[i]):
            return 1
        
def disease(x):     

    for i in range(len(disease_list)):
        if x.__contains__(disease_list[i]):
            return 1
        
def accident(x):         

    for i in range(len(accident_list)):
        if x.__contains__(accident_list[i]):
            return 1


df.good_behavior = text_df.specialmark.map(good_behavior)
df.sensitive = text_df.specialmark.map(sensitive)
df.disease = text_df.specialmark.map(disease)
df.accident = text_df.specialmark.map(accident)

# -

df.iloc[:5, :]

df.to_csv('./EDA_data', index=False)
