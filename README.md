# ML_Portfolio
### 머신러닝 프로젝트 포트폴리오 정리
***
## Team Project 2 - 유기동물 입양 가능성 예측 및 입양 활성화를 위한 DATA 활용 방안
+ Background

반려동물 시장 성장과 함께 건강한 반려동물 문화 형성은 증가하고 있지만 그에 반해 유기동물의 수는 해마다 빠르게 증가하고 유기동물 입양률은 매년 하락하면서, 유기된 동물들을 강제 안락사의 위험과 보호소의 열악한 환경에서 방치되는 현실에서 보호하고자 이미 유기된 동물의 입양 활성화 방안에 포커스<br>             **유기동물 입양 가능성 예측 모델을 통해 유기견, 유기묘의 입양 활성화 제언 및 DATA 활용 방안 제시**

+ Tools : Python (Jupyter notebook, Pycharm)
 
+ Environment : MacOS, Ubuntu Linux, Window

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
    - Plot_importance : 중요 피처 시각화<br>
    - Threshold 조정<br>
