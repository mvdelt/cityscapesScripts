# i.21.3.29.0:29) evalPixelLevelSemanticLabelingJ.py 에서 내플젝에맞게끔 args.avgClassSize 값 계산해주기위한 파일.
#  idea: 걍 통상적인 방법대로, ~~instanceIds.png 를 읽어들여서 넘파이로 만든담에,
#  np.unique 사용해서 area들 구하고 평균내주면 될듯.
#  5000, 5001, 5002,   6000, 6001, ...  이런식이니까, 
#  모든 val 이미지들에 대해서 각 클래스별로 싹다 area 합해준담에
#  인스턴스 갯수로 나눠주면 되겠지.
#  TODO 지금자야해서 낼 하자.


# i.21.3.31.21:52) 어젠가 args.avgClassSize 가 뭔지 확실히 알았으니 이제 계산해주자. 
#  각 클래스의 (gt)인스턴스의 area 평균임. 
#  변수명을 다시짓는다면 classAvgInsSize 정도의 변수명이 맞겠네. 
#  코랩상에서 실행시켜주는걸 가정하고 경로 정해준거임.

import glob
import numpy as np
from PIL import Image 
# i.21.4.1.1:28) TODO Q: 내컴에선 일케 임포트하면 되는데, 코랩에서 임포트하면 일케하면 못찾고 밑밑줄처럼 임포트해줘야 인식하네. 왜지??? 
from cityscapesscripts.helpers.labels import labels, id2label
# from cityscapesScripts.cityscapesscripts.helpers.labels import labels, id2label


# MYROOTDIRPATH_J = "/content/datasetsJ/panopticSeg_dentPanoJ"
searchStr = "/content/datasetsJ/panopticSeg_dentPanoJ/gt/val/*_instanceIds.png" # i. 코랩컴에서의 경로. val 데이터셋에대해서만 해주는거겠지?? /21.3.31.22:29. 
gtList = glob.glob(searchStr)


# i.21.3.31.23:56) 방법 1. 뭔가 되게 좀 별로인 방법같음. 최적화안된듯한. 지금 넘늦은시간이라그런가 머리가안돌아감... 
areaSum_and_insCnt = {
    # 't_normal': [23522, 4],
    # ...
}
for label in labels:
    if not label.hasInstances: # i. ignoreInEval 값 체크는 내플젝에선 걍 패스하자 어차피 다 False 이니. 
        continue
    areaSum_and_insCnt[label.name] = [0, 0]
    for fpath in gtList:
        gtImg = Image.open(fpath)
        gtArr = np.array(gtImg)
        # areaSum
        gtArr_labelIds = gtArr//1000 # i. thing 들의 labelId들만 남고, stuff 등 나머지애들은 0 이 됨. 
        labelIds, areaSums = np.unique(gtArr_labelIds, return_counts=True) # labelIds 에는 thing 들의 labelId들만 있고, 나머지값들은 0. 
        print(f'j) areaSums: {areaSums}') 
        idx = np.where(labelIds==label.id)[0][0]
        print(f'j) idx: {idx}')
        areaSum = areaSums[idx]
        print(f'j) areaSums[idx]: {areaSums[idx]}')
        areaSum_and_insCnt[label.name][0] += areaSum
        # insCnt
        insIds = np.unique(gtArr)
        labelIds, insCnts = np.unique(insIds//1000, return_counts=True) 
        idx = np.where(labelIds==label.id)
        insCnt = insCnts[idx]
        areaSum_and_insCnt[label.name][1] += insCnt
avgClassSizeJ = {}
for labelName, asum_inscnt in areaSum_and_insCnt.items():
    avgClassSizeJ[labelName] = asum_inscnt[0]/asum_inscnt[1]
print(f'j) avgClassSizeJ   by 방법1: {avgClassSizeJ}')


# i.21.3.31.23:55) 방법 2. 
areaSum_and_insCnt_2 = {
    # 't_normal': [23522, 4],
    # ...
}
for label in labels:
    if label.hasInstances:
        areaSum_and_insCnt_2[label.name] = [0, 0] 
for fpath in gtList:
    gtImg = Image.open(fpath)
    gtArr = np.array(gtImg)
    insIds, insAreas = np.unique(gtArr, return_counts=True)
    for insId, insArea in zip(insIds, insAreas):
        if insId//1000 == 0:
            continue
        labelName = id2label[insId//1000].name
        areaSum_and_insCnt_2[labelName][0] += insArea 
        areaSum_and_insCnt_2[labelName][1] += 1
avgClassSizeJ_2 = {}
for labelName, asum_inscnt in areaSum_and_insCnt_2.items():
    avgClassSizeJ_2[labelName] = asum_inscnt[0]/asum_inscnt[1]
print(f'j) avgClassSizeJ_2 by 방법2: {avgClassSizeJ_2}')



# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color(RGB)
#     Label(  'unlabeled_Label'      ,  0 ,        0,  'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
#     Label(  'mandible'             ,  1 ,        1 , 'boneJ'           , 1       , False        , False        , (135,128,255) ),
#     Label(  'maxilla'              ,  2 ,        2 , 'boneJ'           , 1       , False        , False        , (207,221,255) ),
#     Label(  'sinus'                ,  3 ,        3 , 'sinusJ'          , 2       , False        , False        , (  0,  0,255) ),
#     Label(  'canal'                ,  4 ,        4 , 'canalJ'          , 3       , False        , False        , (255,  0,  0) ),  # i. canal 이 젤 고난이도니까, 나중에 hasInstances True로도 실험해보자 어찌되는지.
#     Label(  't_normal'             ,  5 ,        5 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  6 ,        6 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
# ]