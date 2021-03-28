# i.21.3.29.0:29) evalPixelLevelSemanticLabelingJ.py 에서 내플젝에맞게끔 args.avgClassSize 값 계산해주기위한 파일.
#  idea: 걍 통상적인 방법대로, ~~instanceIds.png 를 읽어들여서 넘파이로 만든담에,
#  np.unique 사용해서 area들 구하고 평균내주면 될듯.
#  5000, 5001, 5002,   6000, 6001, ...  이런식이니까, 
#  모든 val 이미지들에 대해서 각 클래스별로 싹다 area 합해준담에
#  인스턴스 갯수로 나눠주면 되겠지.
#  TODO 지금자야해서 낼 하자.
