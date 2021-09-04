KoElectra + CRF (Named Entity Recognition, NER)    
=============
2021 Graduation Project
-------------
--------
## 주제
### *텍스트 내 지리정보 추출 패키지 제작*   
( 현재 웹 페이지 내에 존재하는 지리정보를 하이라이팅 하는 크롬 익스텐션 )

----
## 모델
[한국어로 Pre-training 된 Electra](https://github.com/monologg/KoELECTRA) 에 [CRF layer](https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html#CRF) 
를 추가하여 NER task 를 수행

----
## 사용
인자로 넘어온 문장들에 대해 NER task 수행 후 
Location tag 의 entity 만 추출 하고
Konlpy 의 Okt(Open Korean text) 를 사용하여
해당 entity 의 조사를 제거해 서버로 반환한다.   
*(60_final/ner.py의 NER 함수)*

--- 
## 정리
[과정](https://www.notion.so/KoELECTRA-CRF-c9a5ac67cf7d4c25a15eb9d1026a16b9#2923ec4d8e0e41c3bdd7c17a10205db3)   
[결과](https://www.notion.so/c5b7d10122f341ce922b224df2449ea3?v=4b032969dd224478aa027632046e2e03)   
[결론](https://www.notion.so/KoELECTRA-CRF-c9a5ac67cf7d4c25a15eb9d1026a16b9#ed086da650b2412888b51afea92663c4)   
[참고자료](https://www.notion.so/7669700ad0d04138a4a476664d2e658c)













