# ML_Portfolio
### 머신러닝 프로젝트 포트폴리오 정리
***
### Team Project 1
## 롯데의 고객은 누구일까? 페르소나 찾기 _제5회 L.POINT Big Data Competition 데이터 

+ Tools : SQL (Oracle) / Python (Jupyter notebook)
 
+ Environments : MacOS, Ubuntu Linux, Window

+ Background


+ Summary

  (1). Data Collection<br>
    - 수집대상 : 롯데그룹 온라인 계열사의 온라인행동 데이터 및 상품분류 데이터<br>
    - 수집기간 : 2018.04 ~ 2019.09 총 6개월<br>
    - 수집건수 : <br>
    - 수집출처 : 제5회 L.POINT Big Data Competition "Be the L.BA" 제공<br>
   
   (2). Data Preprocessing<br>
     - 이상치/결측치 처리


### Team Project 2
## 유기동물 입양 가능성 예측 및 입양 활성화를 위한 DATA 활용 방안

+ Tools : Python (Jupyter notebook)
 
+ Environments : MacOS, Ubuntu Linux, Window

+ Background

반려동물 시장 성장과 함께 건강한 반려동물 문화 형성은 증가하고 있지만 그에 반해 유기동물의 수는 해마다 빠르게 증가하고 유기동물 입양률은 매년 하락하면서, 유기된 동물들을 강제 안락사의 위험과 보호소의 열악한 환경에서 방치되는 현실에서 보호하고자 이미 유기된 동물의 입양 활성화 방안에 포커스<br>             **유기동물 입양 가능성 예측 모델을 통해 유기견, 유기묘의 입양 활성화 제언 및 DATA 활용 방안 제시**

+ Summary

  (1). Data Collection<br>
    - 수집대상 : 유기, 유실 강아지 및 고양이 공고/관리현황 데이터 (유기번호/발견장소/품종/색상/성별/특징/나이 등 20개 변수)<br>
    - 수집기간 : 2017.01.01 ~ 2020.10.12<br>
    - 수집건수 : 324,702 건<br>
    - 수집방법 : 오픈 API 활용<br>
    - 수집출처 : 농림축산검역본부 동물보호관리시스템 유기동물 조회 서비스<br>

  (2). Data Preprocessing<br>
    - 이상치/결측치 처리<br>
    - 데이터 변수 성격에 따른 타입 변환<br>
    - 데이터 범주화 (Categorizing/Factoring)<br>
    - 비정형 데이터 텍스트 전처리<br>
      정규표현식 / 불용어 제거 / 워드 토큰화(한글 형태소 분석기 Mecab 적용) / 단어집합 해시맵 Vocab 생성<br>

  (3). Model & Algorithms<br>
    - Word2Vec (Skip-gram) : 워드 임베딩<br>
    - Kemans Clustering : 토픽 워드 도출<br>
    - WordCloud : 텍스트 분석 시각화<br>
    - Light GBM : 분류 분석 모델 (교차 검증으로 분류 모델 점수 비교 및 모델 최종 선정)<br>
    - GridSearchCV : 하이퍼파라미터 조정<br>
    - Feature importance : 중요 피처 시각화<br>
    - Threshold 조정<br>

+ Conclusion

  (1). 분류모델 Light GBM / Decision Tree Classifier / Logistic Regression / Random Forest Classifier / Gradient boosting / XGboost / GaussianNB / K neighbors Classifier / Catboost 분석 결과 Light GBM의 f1_score, roc-score 가 가장 높게 나옴<br>
  
  (2). Light GMB 모델 결정 후 k-fold 교차검증 및 GridSearchCV로 하이퍼 파라미터 조정을 통해 score를 높이는 과정 진행<br>
  
  (3). Feature importance를 통해 분석 결과에 큰 영향을 준 주요 변수 도출 (공고일자/몸무게/유기된장소/성별/나이/특이사항)<br>
  
  (4). 분석 결과 및 제언<br>
    - 7~10월 반려동물 유기 증가 기간의 집중 캠페인 시행<br>
    - 보호소 신규 개설 보다 직접적인 영향을 미치는 제도 마련<br>
    - 질병 및 사고로 입양 가능성이 낮은 유기동물의 질병 치료 우선<br>
 
+ Further Research

  (1). 데이터의 한계<br>
    - 공고 기간이나 유기동물 입양 여부까지만 확인 가능<br>
    - 입양된 시점에 대한 데이터 부재<br>
    - 입양된 시점의 데이터가 제공된다면 입양 소요 시간 예측 모델을 통해 더욱 효율적인 유기동물 관리와 입양 활성화 방안 제시가 가능할 것으로 보임<br>

  (2). 분석 기법의 한계<br>
    - 입양에 중요한 영향을 미칠것이라고 예상되는 이미지를 사용한 딥러닝 모델링 미적용<br>
    - 딥러닝을 통한 정확한 분류 예측 모델 필요<br>

  (3). 해당 모델링의 타 분야 활용 방안<br>
    - 물류 시스템 분류 예측 모델 적용 가능<br>
