
# coding: utf-8

# # 롯데의 고객은 누구일까? 페르소나 찾기_ 롯데 온라인 고객 분석
# ### 제 5회 L.POINT Big Data Competition
# #### 온라인 행동 데이터 기반으로 구매자의 특성을 분류하여 롯데 그룹 온라인 계열사의 페르소나를 선정하고 시각화를 통한 인사이트 도출
# #### -> 롯데의 온라인 구매 고객 핵심 타겟층의 니즈와 특성을 파악해 향후 마케팅 전략에 반영 가능한 분석 자료로 활용 가능

# ## 1. Library & Data Import

# In[208]:


import pandas as pd
import numpy as np
from pprint import pprint as pp
from inspect import signature
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action = 'ignore')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
 
get_ipython().system('apt -qq -y install fonts-nanum')
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()


# ## 2. EDA 탐색적 데이터 분석
# ### 2-1. 데이터 셋 기본 정보 파악

# #### Law Data Set Summary : 분석 데이터 구성 _ 온라인 행동 데이터 영역 & 상품 분류 체계 영역
# - Custom : 고객 정보 (고객의 인구통계학적 특징)
# - Session : 세션 (고객 온라인 행동 데이터)
# - Product : 주문 (상품 특징별 주문 금액, 횟수)
# - Master : 상품 정보 (상품 분류별 고객 니즈)
# - Search1 : 검색어1 (검색&구매 상관관계, 키워드)
# - Search2 : 검색어2 (분석 데이터 기간 내 일별&검색어별 검색량)

# ### 2-1-1. Custom Data

# In[209]:


df_cus = pd.read_csv('./Custom.csv')


# In[210]:


pp( df_cus.info() )


# In[211]:


df_cus.head()


# In[212]:


#고객 정보 데이터 기초 통계

#고객 연령대별 및 성별 볼륨 파악

print(df_cus['CLNT_AGE'].value_counts())
print('---------------------------------------')
print(df_cus['CLNT_GENDER'].value_counts())


# In[213]:


#고객 연령대에 따른 성별 볼륨 파악

df_ages = df_cus.groupby( ['CLNT_AGE', 'CLNT_GENDER'] )['CLNT_GENDER'].count()
df_ages


# In[214]:


#데이터 시각화
#고객 연령대 확인

sns.set_style('whitegrid')

grid = plt.subplots() 

grid = sns.countplot('CLNT_AGE', data=df_cus, palette='coolwarm')
grid.set_title('Total LOTTE ONLIEN CUSTOMER Age Histogram')
grid.set_xlabel('LOTTE ONLIEN CUSTOMER Age')

plt.show()


# In[215]:


#데이터 시각화
#고객 성별 비중 확인

grid = plt.subplots()

grid = sns.boxplot(x='CLNT_GENDER', y='CLNT_AGE', data=df_cus)
grid.set_title('Boxplot of LOTTE ONLIEN CUSTOMER Age by Gender')
grid.set_xlabel('gender')
grid.set_ylabel('age')

plt.show()


# In[216]:


#데이터 시각화
#고객 성별 비중 확인

grid = plt.subplots()

grid = sns.violinplot(x='CLNT_GENDER', y='CLNT_AGE', data=df_cus)
grid.set_title('Boxplot of LOTTE ONLIEN CUSTOMER Age by Gender')
grid.set_xlabel('gender')
grid.set_ylabel('age')

plt.show()


# ### 2-1-2. Session Data

# In[217]:


df_sess = pd.read_csv('./Session.csv')


# In[218]:


pp( df_sess.info() )


# In[219]:


df_sess.head()


# ### 2-1-3. Product Data

# In[220]:


df_prod = pd.read_csv('./Product.csv')


# In[221]:


df_prod.info()


# In[222]:


df_prod.head()


# ### 2-1-4. Master Data

# In[223]:


df_mast = pd.read_csv('./Master (2).csv')


# In[224]:


df_mast.info()


# In[225]:


df_mast.head()


# ### 2-1-5. Search1 Data

# In[226]:


df_sch1 = pd.read_csv('./Search1 (2).csv')


# In[227]:


df_sch1.info()


# In[228]:


df_sch1.head()


# ### 2-1-5. Search2 Data

# In[229]:


df_sch2 = pd.read_csv('./Search2.csv')


# In[230]:


df_sch2.info()


# In[231]:


df_sch2.head()


# ## 3. 분석 방향에 적합한 새로운 데이터 셋 생성
# 롯데의 핵심 고객, 즉 구매력이 높은 고객은 누구일까?
# 기준은 어떻게 잡아야 할까?
# 
# 구매력이 높은 충성고객 정의 : 총구매금액과 구매횟수 기준으로 4개의 그룹으로 분류
# - SQL 활용 : 고객 세그멘테이션 데이터 셋 생성 

# ## 4. 고객 세그멘테이션 데이터 셋 : df_segs 데이터 기초 탐색

# In[232]:


df_segs = pd.read_csv('./segs.csv')


# In[233]:


pp( df_segs.info() )


# In[234]:


df_segs.head()


# #### -> 그룹을 이해하는 차원에서 각 그룹별 흥미롭게 기초 통계 살펴봄
# ### 4-1. df_segs 데이터 기초 통계량 
# 최소의 금액을 소비한 고객 & 구매횟수 제일 적은 고객

# In[235]:


pp( df_segs.min() ) # 각 변수의 최솟값 확인
pp('-'*30)
pp( df_segs['PD_BUY_TOT'].min() ) # 최소 금액 산출
pp('-'*30)
pp( df_segs['BUY_COUNT'].min() ) # 제일 적은 구매횟수 산출


# In[236]:


# 가장 적은 금액을 소비한 고객은 누구일까? a.k.a 짠돌이

df_segs[df_segs['PD_BUY_TOT'] == df_segs['PD_BUY_TOT'].min()]


# In[237]:


# 가장 적은 횟수로 롯데 온라인 플랫폼을 이용한 고객은 과연 누구일까?

df_segs[df_segs['BUY_COUNT'] == df_segs['BUY_COUNT'].min()]


# In[238]:


# 어메이징한 금액을 소비한 고객 & 구매횟수가 제일 많은 고객은 누구일까?

pp( df_segs.max() ) # 각 변수의 최댓값 확인
pp('-'*30)
pp( df_segs['PD_BUY_TOT'].max() ) # 최대 구매 금액 산출
pp('-'*30)
pp( df_segs['BUY_COUNT'].max() ) # 제일 많은 구매횟수 산출


# ### 4-2. df_segs 데이터 그룹별 새 데이터프레임 생성
# #### Group A (CUS_SEG 1)
# 총구매횟수는 평균보다 적지만 총구매금액이 평균보다 높게 형성된 고객 그룹

# In[239]:


Agroup = df_segs[ df_segs['CUS_SEG'] == 1 ]
Agroup


# #### Group B (CUS_SEG 2)
# 총구매횟수가 평균 이상이고 총구매금액도 평균 이상인 매우 이상적인 고객 그룹

# In[240]:


Bgroup = df_segs[ df_segs['CUS_SEG'] == 2 ]
Bgroup


# #### Group C (CUS_SEG 3)
# 총구매횟수가 평균 이하이고 총구매금액도 평균 이하인 고객 그룹

# In[241]:


Cgroup = df_segs[ df_segs['CUS_SEG'] == 3 ]
Cgroup


# #### Group D (CUS_SEG 4)
#  총구매횟수가 평균 이상인데 총구매금액은 평균 이하인 고객 그룹

# In[242]:


Dgroup = df_segs[ df_segs['CUS_SEG'] == 4 ]
Dgroup


# #### df_cus & df_segs 데이터프레임 Inner join하여 새 데이터프레임 생성
# 고객 연령대 및 성별 정보가 있는 df_cus 기준으로 고객 그룹 분리한 df_segs 데이터프레임 조인

# In[243]:


df_join = pd.merge(left = df_cus, right = df_segs, how = 'inner', on = 'CLNT_ID')
df_join


# #### df_join 데이터 그룹별 새 데이터프레임 생성
# #### Group 1 (CUS_SEG 1)
# 총구매횟수는 평균보다 적지만 총구매금액이 평균보다 높게 형성된 고객 그룹

# In[244]:


group1 = df_join[ df_join['CUS_SEG'] == 1 ]
group1


# In[245]:


group1.info()


# #### Group 2 (CUS_SEG 2)
# 총구매횟수가 평균 이상이고 총구매금액도 평균 이상인 매우 이상적인 고객 그룹

# In[246]:


group2 = df_join[ df_join['CUS_SEG'] == 2 ]
group2


# In[247]:


group2.info()


# #### Group 3 (CUS_SEG 3)
# 총구매횟수가 평균 이하이고 총구매금액도 평균 이하인 고객 그룹

# In[248]:


group3 = df_join[ df_join['CUS_SEG'] == 3 ]
group3


# In[249]:


group3.info()


# #### Group 4 (CUS_SEG 4)
# 총구매횟수가 평균 이상인데 총구매금액은 평균 이하인 고객 그룹

# In[250]:


group4 = df_join[ df_join['CUS_SEG'] == 4 ]
group4


# In[251]:


group4.info()


# ## 5. 데이터 시각화 & 인사이트
# ### 각 그룹별 고객 파악하기
# 롯데의 주 고객층은 어떤 사람일까?

# In[252]:


# 그룹1

grid = plt.subplots()

grid = sns.countplot('CLNT_AGE', data=group1 , hue='CLNT_GENDER')

grid.set_title('LOTTE ONLIEN CUSTOMER GROUP 1')
grid.set_xlabel('Age')
grid.set_ylabel('Count')

plt.show()


# In[253]:


# 그룹2

grid = plt.subplots()

grid = sns.countplot('CLNT_AGE', data=group2 , hue='CLNT_GENDER')

grid.set_title('LOTTE ONLIEN CUSTOMER GROUP 2')
grid.set_xlabel('Age')
grid.set_ylabel('Count')

plt.show()


# In[254]:


# 그룹3

grid = plt.subplots()

grid = sns.countplot('CLNT_AGE', data=group3 , hue='CLNT_GENDER')

grid.set_title('LOTTE ONLIEN CUSTOMER GROUP 3')
grid.set_xlabel('Age')
grid.set_ylabel('Count')

plt.show()


# In[255]:


# 그룹4

grid = plt.subplots()

grid = sns.countplot('CLNT_AGE', data=group4 , hue='CLNT_GENDER')

grid.set_title('LOTTE ONLIEN CUSTOMER GROUP 4')
grid.set_xlabel('Age')
grid.set_ylabel('Count')

plt.show()


# - 모든 그룹에서 30대 여성, 40대 여성, 20대 여성 순으로 구매자 수가 가장 많았음을 확인

# ### 각 그룹별 고객 파악하기
# 총구매금액 기준, 구매력이 있는 고객은 남성일까 여성일까?

# In[256]:


# 전체 고객에서 남성과 여성의 총구매금액 새로운 데이터프레임 생성

cusbuytot = df_join[['CLNT_ID', 'CLNT_GENDER', 'PD_BUY_TOT']]

cusbuytot.head()


# In[257]:


cusbuytot.info()


# In[258]:


print(cusbuytot[cusbuytot['CLNT_GENDER'] == 'F'].sum())
print('----------------------------------------------------------')
print(cusbuytot[cusbuytot['CLNT_GENDER'] == 'M'].sum())


# In[259]:


# 전체 고객에서 총구매금액 기준 구매력 있는 성별 확인 파이 차트

labels = 'Female', 'Male'
sizes = [179794063190, 31235932801 ]
explode = (0, 0)
colors = ['lightcoral', 'mediumaquamarine']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')

sns.set(font_scale=1.305)

plt.show()


# In[260]:


# 각 그룹별 총구매금액 기준 구매력 있는 성별 확인
# 그룹1

print(group1[group1['CLNT_GENDER'] == 'F'].sum())
print('----------------------------------------------------------')
print(group1[group1['CLNT_GENDER'] == 'M'].sum())


# In[261]:


# 그룹1 총구매금액 기준 구매력 있는 성별 확인 파이 차트

labels = 'Female', 'Male'
sizes = [34058297010, 9647924730]
explode = (0, 0)
colors = ['lightcoral', 'mediumaquamarine']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('GROUP 1')

sns.set(font_scale=1.305)

plt.show()


# In[262]:


# 각 그룹별 총구매금액 기준 구매력 있는 성별 확인
# 그룹2

print(group2[group2['CLNT_GENDER'] == 'F'].sum())
print('----------------------------------------------------------')
print(group2[group2['CLNT_GENDER'] == 'M'].sum())


# In[263]:


# 그룹2 총구매금액 기준 구매력 있는 성별 확인 파이 차트

labels = 'Female', 'Male'
sizes = [122281811470, 17502953121]
explode = (0, 0)
colors = ['lightcoral', 'mediumaquamarine']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('GROUP 2')

sns.set(font_scale=1.305)

plt.show()


# In[264]:


# 각 그룹별 총구매금액 기준 구매력 있는 성별 확인
# 그룹3

print(group3[group3['CLNT_GENDER'] == 'F'].sum())
print('----------------------------------------------------------')
print(group3[group3['CLNT_GENDER'] == 'M'].sum())


# In[265]:


# 그룹3 총구매금액 기준 구매력 있는 성별 확인 파이 차트

labels = 'Female', 'Male'
sizes = [17469200090, 3112859180]
explode = (0, 0)
colors = ['lightcoral', 'mediumaquamarine']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('GROUP 3')

sns.set(font_scale=1.305)

plt.show()


# In[266]:


# 각 그룹별 총구매금액 기준 구매력 있는 성별 확인
# 그룹4

print(group4[group4['CLNT_GENDER'] == 'F'].sum())
print('----------------------------------------------------------')
print(group4[group4['CLNT_GENDER'] == 'M'].sum())


# In[267]:


# 그룹4 총구매금액 기준 구매력 있는 성별 확인 파이 차트

labels = 'Female', 'Male'
sizes = [5984754620, 972195770]
explode = (0, 0)
colors = ['lightcoral', 'mediumaquamarine']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('GROUP 4')

sns.set(font_scale=1.305)

plt.show()


# - 여성이 85% 정도로 압도적으로 높음 
# - 여성의 소비액 또한 남성보다 많음을 확인 할 수 있었음(그룹1인 경우, 남성의 비중이 그룹 2보다 최대 10%p나 차이남)
# - 다른 그룹에 비해 남성의 비율이 상대적으로 높았기에 남성의 소비가 차지하는 비중이 다른 그룹보다 높게 형성되어 있음
# - 전체적으로 롯데의 주고객이 여성인 것을 알 수 있었음

# ### 각 그룹별 고객 파악하기
# 구매횟수 기준, 구매력이 있는 고객은 남성일까 여성일까?

# In[268]:


# 전체 고객에서 남성과 여성의 총구매횟수 새로운 데이터프레임 생성

cusbuyc = df_join[['CLNT_ID', 'CLNT_GENDER', 'BUY_COUNT']]

cusbuyc.head()


# In[269]:


print(cusbuyc[cusbuyc['CLNT_GENDER'] == 'F'].sum())
print('---------------------------------------------')
print(cusbuyc[cusbuyc['CLNT_GENDER'] == 'M'].sum())


# In[270]:


# 전체 고객에서 구매횟수기준 구매력있는 성별 확인 파이차트

labels = 'Female', 'Male'
sizes = [3490390, 496885]
explode = (0, 0)
colors = ['coral', 'slateblue']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')

sns.set(font_scale=1.305)

plt.show()


# In[271]:


# 각 그룹별 구매횟수 기준 구매력 있는 성별 확인 파이차트
# 그룹1

labels = 'Female', 'Male'
sizes = [211296, 47924]
explode = (0, 0)
colors = ['coral', 'slateblue']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('Group 1')

sns.set(font_scale=1.305)

plt.show()


# In[272]:


# 각 그룹별 구매횟수 기준 구매력 있는 성별 확인 파이차트
# 그룹2

labels = 'Female', 'Male'
sizes = [2405418, 293911]
explode = (0, 0)
colors = ['coral', 'slateblue']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('Group 2')

sns.set(font_scale=1.305)

plt.show()


# In[273]:


# 각 그룹별 구매횟수 기준 구매력 있는 성별 확인 파이차트
# 그룹3

labels = 'Female', 'Male'
sizes = [535490, 97797]
explode = (0, 0)
colors = ['coral', 'slateblue']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('Group 3')

sns.set(font_scale=1.305)

plt.show()


# In[274]:


# 각 그룹별 구매횟수 기준 구매력 있는 성별 확인 파이차트
# 그룹4

labels = 'Female', 'Male'
sizes = [338186, 57253]
explode = (0, 0)
colors = ['coral', 'slateblue']

frame, grid = plt.subplots()
grid.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
grid.axis('equal')
plt.title('Group 4')

sns.set(font_scale=1.305)

plt.show()


# - 여성이 롯데 온라인 플랫폼에서 소비하는 횟수 또한 남성보다 많음을 확인 87.5% 

# ### 각 그룹별 고객 파악하기
# 고객들은 어떤 디바이스로 롯데 온라인 쇼핑을 즐길까?

# In[275]:


# 결측치 확인 및 삭제
# 결측치 개수 확인

df_sess.isnull().sum()


# In[276]:


# 결측치 데이터 제외

df_sess = df_sess.dropna()


# In[277]:


df_sess = df_sess.merge(df_segs, left_on='CLNT_ID', right_on='CLNT_ID')


# In[278]:


# 1. datetime 변환
#데이터 형 변환 - 실패 

df_sess.SESS_DT.astype('datetime64[ns]')


# In[279]:


# 문자열로 우선 바꾸고, 날짜로 인식가능하게 문자열 규칙 정해주기   

df_sess.SESS_DT = df_sess.SESS_DT.astype(str) # str로 바꿀수 있음...
df_sess.SESS_DT.dtype


# In[280]:


# 세션일자 형변환 완료

df_sess.SESS_DT = df_sess.SESS_DT.astype('datetime64[ns]')
df_sess.info()


# In[281]:


# 2. float 타입으로 변경 (결측치 때문에 일괄 float로 변경 필요)

df_sess.TOT_PAG_VIEW_CT.astype(float)
df_sess.TOT_SESS_HR_V = df_sess.TOT_SESS_HR_V.str.replace(',','').astype(float)


# - 전체 고객 및 그룹별 디바이스 사용 비중 확인

# In[282]:


# 관측치 값 확인

df_sess.DVC_CTG_NM.unique()


# In[283]:


mobile = (df_sess[df_sess.DVC_CTG_NM == 'mobile'].count() / df_sess.DVC_CTG_NM.count() * 100)[0]
desktop = (df_sess[df_sess.DVC_CTG_NM == 'desktop'].count() / df_sess.DVC_CTG_NM.count() * 100)[0]
tablet = (df_sess[df_sess.DVC_CTG_NM == 'tablet'].count() / df_sess.DVC_CTG_NM.count() * 100)[0]


# In[284]:


pp(mobile)
pp(desktop)
pp(tablet)


# In[285]:


# 전체 구매자의 디바이스 사용 비중

frame, grid = plt.subplots()

labels = 'desktop', 'mobile', 'tablet'
sizes = [desktop, mobile, tablet]
explode = (0, 0.1, 0)
grid.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title("Device Ratio")
plt.show()


# In[286]:


# 그룹별 디바이스 사용 비중 바 차트 

grouped_device = df_sess.groupby(['DVC_CTG_NM','CUS_SEG'])['CUS_SEG'].count()

labels = ['1', '2', '3', '4']
mobile = grouped_device['mobile']
desktop = grouped_device['desktop']
tablet = grouped_device['tablet']

width = 0.8        

fig, ax = plt.subplots()

ax.bar(labels, mobile, width, label='mobile')
ax.bar(labels, desktop, width,  bottom=mobile,label='desktop')
ax.bar(labels, tablet, width,  bottom=desktop,label='tablet')

ax.set_title('Device Usages by Segments')
ax.set_xlabel('Groups')
ax.set_ylabel('persons')

ax.legend()

plt.show()


# In[287]:


# 그룹별 디바이스 사용 수 - 100% 비율로 보기 
# 데이터 프레임 만들기

df = pd.DataFrame(df_sess.groupby('CUS_SEG')['CUS_SEG'].count())
df['mobile'] = grouped_device['mobile'] / df.CUS_SEG * 100 
df['desktop'] = grouped_device['desktop'] / df.CUS_SEG * 100 
df['tablet'] = grouped_device['tablet'] / df.CUS_SEG * 100 
df


# In[288]:


labels = ['1', '2', '3', '4']
mobile = df.mobile
desktop = df.desktop
tablet = df.tablet

width = 0.8        

fig, ax = plt.subplots()


ax.bar(labels, desktop, width,  label='desktop')
ax.bar(labels, mobile, width, bottom=desktop, label='mobile')
ax.bar(labels, tablet, width, bottom=desktop, label='tablet')


ax.set_title('Device Usages by Segments')
ax.set_xlabel('Groups')
ax.set_ylabel('persons')

ax.legend()

plt.show()


# - 모바일>데스크탑>태블릿 순으로 접속 비율이 높음을 확인
# - → 모바일 기기의 보편화 등 여러 가능한 이유가 있겠지만 롯데 고객군은 모바일로 언제, 어디서나 편리하게 쇼핑을 하고자 하는 그룹으로 예상
# - → 롯데가 모바일 쇼핑 환경을 만드려는 데에 좋은 환경이 마련되어 있음을 유추  
# - 파이차트는 그룹별 크기를 바로 비교하기에는 쉽지 않아서, 그룹바 차트를 100%로 늘려서 확인해 본 결과 그룹2에서 모바일 사용 비중이 가장 높음을 확인
# - → 시간과 장소의 제약 없이 쇼핑을 즐기며 쇼핑을 자주하는 그룹으로 형성되어 있을 것으로 예상

# ### 각 그룹별 고객 파악하기
# 재구매율은 얼마나 될까? (재방문 구매자 파악)

# In[289]:


sess_seq_min = df_sess.groupby('CLNT_ID')['SESS_SEQ'].min()
sess_seq_min


# In[290]:


# 재방문자 행추출 
r = df_sess[df_sess['SESS_SEQ'] > 1 ]
revisitors = r.groupby('CLNT_ID')['SESS_SEQ'].min()

# 방문자 행추출 
v = df_sess[df_sess['SESS_SEQ'] <= 1 ]
visitors = v.groupby('CLNT_ID')['SESS_SEQ'].min()

# 신규/재방문자 수 변수 정의 
visitors = visitors.count()
revisitors = revisitors.count()


# In[291]:


#신규/재방문율 그래프 그리기 - pie chart

frame, grid = plt.subplots()

labels = 'visitors', 'revisitors'
sizes = [visitors, revisitors]

grid.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title("Revisitors Ratio")

plt.show()


# - 데이터 수집 기간 내 재구매한 고객의 비중이 전체 평균 기준 90%대여서 매우 높은 수치임을 확인 
# - 엘포인트를 사용하는 롯데 고객은 재방문 비율이 높아 어느정도 서비스에 만족한다고 볼 수 있음 → 그러나, 신규 사용자의 비율이 너무 낮아 신규 사용자의 유입을 높일 필요가 있어 보임

# ### 구매 행동 패턴 발견하기
# 고객들은 한 번 구매할 때 얼마나 오랫동안 쇼핑할까?
# (그룹별 평균 세션 시간)

# In[297]:


# 클라이언트별 평균 세션시간 = sh_cid
sh_cid = df_sess.groupby('CLNT_ID')['TOT_SESS_HR_V'].mean()
sh_cid = sh_cid.reset_index()
sh_cid = sh_cid.merge(df_sess, left_on='CLNT_ID', right_on='CLNT_ID', suffixes=('_x', '_y'))


# In[298]:


# 분단위로 변경 
sh_cid['TOT_SESS_HR_V_x'] = sh_cid.TOT_SESS_HR_V_x/60  


# In[299]:


# 데이터 확인 
sh_cid.head()


# In[300]:


#그룹별 및 전체 평균 PV 조회수 그래프 그리기

frame, grid = plt.subplots()

grid = sns.barplot(x='CUS_SEG', y='TOT_SESS_HR_V_x', data=sh_cid, palette='summer')
grid = plt.axhline(y=sh_cid.TOT_SESS_HR_V_x.mean(), color='r', linewidth=1)

plt.title("Average Session Minute by Customer Segments")

plt.show()


# - 전체 평균은 24.19분이었고, 구매횟수가 높은 그룹이 평균 이상으로 나와 세션 시간 또한 구매횟수가 높은 그룹과 관련있어 보임
# - 그룹1과 3에서 세션 시간이 평균 이하인 것으로 보아 온라인 행동이 적극적이지 않음을 파악 =>네이버와 같은 다른 사이트에서 검색을 주로 하다가 롯데몰에 들어와서 구매만 하는 그룹으로 구성되어 있지 않을까?
# - 반면, 그룹2와 그룹4는 세션시간이 평균 이상인 것으로 보아 탐색적 온라인 행동을 많이 한 것으로 확인
# - 전체 평균은 24분으로, 롯데 고객은 한번 방문해서 구매할 때 약 24분 정도의 쇼핑을 하는 것을 알 수 있음 
# - =>구매횟수가 높은 그룹이 평균 이상인 결과가 나와서, 쇼핑에 매우 적극적인 그룹으로 유추할 수 있음
# 
# 

# ### 구매 행동 패턴 발견하기
# 구매자는 평균 몇 페이지를 탐색할까?
# (그룹별 평균 페이지뷰 조회수 파악)

# In[301]:


# 클라이언트별 평균 pv 조회수 = pv_cid
pv_cid = df_sess.groupby('CLNT_ID')['TOT_PAG_VIEW_CT'].mean()
pv_cid = pv_cid.reset_index()
pv_cid = pv_cid.merge(df_sess, left_on='CLNT_ID', right_on='CLNT_ID', suffixes=('_x', '_y'))


# In[302]:


# 조인 결과 확인 
pv_cid.columns


# In[303]:


#그룹별 및 전체 평균 PV 조회수 그래프 그리기
frame, grid = plt.subplots()

grid = sns.barplot(x='CUS_SEG', y='TOT_PAG_VIEW_CT_x', data=pv_cid, palette='summer')
grid = plt.axhline(y=pv_cid.TOT_PAG_VIEW_CT_x.mean(), color='r', linewidth=1)

plt.title("Average PV by Customer Segments")

plt.show()


# - 전체 평균은 85.6페이지로, 롯데 고객은 약 85페이지를 약 24분 동안 탐색한 후 구매를 하는 것임
# - 구매 금액과 상관없이 구매횟수가 적은 그룹(그룹1과 3)의 방문당 페이지 조회수가 낮은 것으로 나타나, 페이지 조회수가 높을수록 구매횟수에 연관이 있어보이는 것을 확인 
