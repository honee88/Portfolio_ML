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

# + [markdown] id="ZxQ2mdWLGiqU"
# # 유기견 입양 여부 예측 

# + [markdown] id="d9rVfKDDGiqV"
# ## 1. Data Collecting
# ### - API 활용 

# + [markdown] id="BNN2FkQtGiqW"
# ### 동물보호 

# + id="zj2pBGxMGiqW"
# 개
page_num = 1
abandonment_data = {}

while page_num <= 34144:

    key = '3oGw9WhjddCEAiQyiVg3VSsrZlJC8D70H%2Ffb6DW7%2FeNjRkAUjItge2uRCjWpfdZCH3%2FwrguCcUBkztUIW1a%2FQA%3D%3D' 
    queryParams = 'bgnde=20170101&endde=20201013&upkind=417000'+'&pageNo='+str(page_num)+'&ServiceKey='+key
    url = 'http://openapi.animal.go.kr/openapi/service/rest/abandonmentPublicSrvc/abandonmentPublic?'+queryParams

    req = requests.get(url) 
    html = req.text
    soup = BeautifulSoup(html,'html.parser')
    
    feature_list = ['desertionno', #유기번호
                'popfile', #이미지 
                'happendt', #접수일
                'happenplace', #발견장소
                'kindcd', #품종
                'colorcd', #색상
                'age', #나이
                'weight', #몸무게 
                'noticeno', #공고번호
                'noticesdt', #공고시작일
                'noticeedt', #공고종료일
                'sexcd', #성별
                'neuteryn',#중성화여부
                'specialmark', #특징
                'carenm', # 보호소 이름 
                'orgnm', #관할기관
                'processstate'] #입양/안락사 등 상태

    
    for feature in feature_list:
        feature_data = soup.find_all(feature)
        
        if abandonment_data.get(feature) is None:
            abandonment_data[feature] = [x.text for x in feature_data]
        else: 
            abandonment_data[feature] = abandonment_data[feature] + [x.text for x in feature_data]
            pass # if 
        pass # for 
    
    pass # while  
    
    print(f'{page_num}완료')   
    page_num += 1
  
df_dog = pd.DataFrame.from_dict(abandonment_data, orient='index')
df_dog = df_dog.transpose()


# + id="obItd0cjGiqZ"
# 고양이 
page_num = 1
abandonment_data = {}

while page_num <= 4533:

    key = '3oGw9WhjddCEAiQyiVg3VSsrZlJC8D70H%2Ffb6DW7%2FeNjRkAUjItge2uRCjWpfdZCH3%2FwrguCcUBkztUIW1a%2FQA%3D%3D' 
    queryParams = 'bgnde=20170101&endde=20201013&upkind=417000'+'&pageNo='+str(page_num)+'&ServiceKey='+key
    url = 'http://openapi.animal.go.kr/openapi/service/rest/abandonmentPublicSrvc/abandonmentPublic?'+queryParams

    req = requests.get(url) 
    html = req.text
    soup = BeautifulSoup(html,'html.parser')
    
    feature_list = ['desertionno', #유기번호
                'popfile', #이미지 
                'happendt', #접수일
                'happenplace', #발견장소
                'kindcd', #품종
                'colorcd', #색상
                'age', #나이
                'weight', #몸무게 
                'noticeno', #공고번호
                'noticesdt', #공고시작일
                'noticeedt', #공고종료일
                'sexcd', #성별
                'neuteryn',#중성화여부
                'specialmark', #특징
                'carenm', # 보호소 이름 
                'orgnm', #관할기관
                'processstate'] #입양/안락사 등 상태

    
    for feature in feature_list:
        feature_data = soup.find_all(feature)
        
        if abandonment_data.get(feature) is None:
            abandonment_data[feature] = [x.text for x in feature_data]
        else: 
            abandonment_data[feature] = abandonment_data[feature] + [x.text for x in feature_data]
            pass # if 
        pass # for 
    
    pass # while  
    
    print(f'{page_num}완료')   
    page_num += 1
  
df_cat = pd.DataFrame.from_dict(abandonment_data, orient='index')
df_cat = df_cat.transpose()


# + id="jDqw9kKIGiqb"
# animalShelterSrvc - 보호소 위,경도 데이터 수집

shelter_data = {}
    
for i in range(0,len(df_dog)):

#     key = '3oGw9WhjddCEAiQyiVg3VSsrZlJC8D70H%2Ffb6DW7%2FeNjRkAUjItge2uRCjWpfdZCH3%2FwrguCcUBkztUIW1a%2FQA%3D%3D' 
    queryParams = 'care_nm='+str(df_dog.carenm[i])+'&ServiceKey='+key
    url = 'http://openapi.animal.go.kr/openapi/service/rest/animalShelterSrvc/shelterInfo?'+queryParams

    req = requests.get(url) 
    html = req.text
    soup = BeautifulSoup(html,'html.parser')

    feature_list = ['carenm', #동물보호센터명
                    'lat', 'lng'] #위,경도

    for feature in feature_list:
        feature_data = soup.find_all(feature)
        
        if shelter_data.get(feature) is None:
            shelter_data[feature] = [x.text for x in feature_data]
        else: 
            shelter_data[feature] = shelter_data[feature] + [x.text for x in feature_data]
            pass # if 
        
        pass #for
    
    pass # for 

df_shelter = pd.DataFrame(shelter_data)


# + id="fxHOa0cpGiqc"
# 테이블 조인 
df_dog_final = pd.merge(df_dog, df_shelter, how='left', left_on='carenm', right_on='carenm')

# + id="SrRY1cbgGiqe"
# df_test = pd.concat([df_2500_all, df_2851, df_3500, df_end]).drop_duplicates().reset_index(drop=True)
df_dog_final.to_csv('./df_dog_final', index=False)

# + id="vILZAkLhGiqg"

