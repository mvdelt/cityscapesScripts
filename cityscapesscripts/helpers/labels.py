#!/usr/bin/python
#
# Cityscapes labels
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.   # <-inverse mapping에선 아래 labels리스트의 (동일한 trainId를 가지는 Label들 중)1번째녀석을 사용한다고 되어있지./i.21.3.5.18:58.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    # i.21.3.5.17:19) ignoreInEval 값이 True 면 evaluation(cityscapes 대회서버에서 하는 이밸류에이션)에 반영 안됨!!!
    #  그래서 train시에만 자유롭게 정해서 이용하라고 trainId 가 있는거고, 
    #  요아래 Label 들의 trainId 값들은 대회측에서 일단 기본값으로 정해논건데(맘대로바꿀수있음), 
    #  ignoreInEval 이 True 인 놈들에 대해서는 trainId 값을 255나 -1등으로 해놓은듯.
    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# i.21.3.6.23:52) 내가 새로 정해준 Label들. color 는 coco-annotator 에서내가정해준 색깔과 유사하게 해놔봣음.
#  category 는 일단 대충 정해봣는데.. 이건 어케하든 상관없는거겟지?
#  hasInstances 를 어케해주냐에 따라서 어떻게 달라지는거지..?????????
#  참고로 hasInstances 가 True 면 (<-코멘트 작성하다 말았음..;;)
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled_Label'      ,  0 ,        0 , 'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
#     Label(  'Rt_sinus'             ,  1 ,        1 , 'sinusJ'          , 1       , False        , False        , (  0,  0,255) ),
#     Label(  'Lt_sinus'             ,  2 ,        2 , 'sinusJ'          , 1       , False        , False        , (255,  0,  0) ),
#     Label(  'maxilla'              ,  3 ,        3 , 'boneJ'           , 2       , False        , False        , (162,156,255) ),
#     Label(  'mandible'             ,  4 ,        4 , 'boneJ'           , 2       , False        , False        , (185,181,247) ),
#     Label(  'Rt_canal'             ,  5 ,        5 , 'canalJ'          , 3       , False        , False        , ( 76, 68,212) ),
#     Label(  'Lt_canal'             ,  6 ,        6 , 'canalJ'          , 3       , False        , False        , (194, 37,144) ),
#     Label(  't_normal'             ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  8 ,        8 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  9 ,        9 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
# ]

########### i.21.3.20.18:24) [레퍼런스버전1] 이게 가장 잘 됐던 버전임. 얘만 totIter 6000 까지 돌려주긴 했지만. (트레이닝셋으로 파노3장가지고) #############
#
# i.21.3.16.23:44) 그냥 작업 간단히 하기위해, Det2에 내재된 수평플립(좌우플립) 걍 이용해주려고, Rt Lt 구분 없게 바꿔주려함.
#  만약 Rt Lt 구분 있었으면 수평플립 꺼주고 대신 내가 직접 오그멘테이션해줘야하지(구강계AI대회에서 내가직접 좌우플립 오그멘테이션해준것처럼).
# i.21.3.17.17:38) TODO: sinus, canal 의 hasInstances 를 True 로 해줘야하나???
# i.21.3.17.19:00) TODO: unlabeled_Label 의 trainId 를 255로 해주는게 나은가??? 그리고나서 모든걸 trainId 기준으로 준비해주고..??
#  (난 지금은 id랑 trainId 가 동일해서 그냥 크게 구분없이 해줫을거임. )
#    아니, 걍 unlabeled_Label 자체를 걍 없애버리고 총 클래스수를 8개가 아닌 7개로 해주면 될것같은데?? 
#  panoptic deeplab 에서도 coco 는 133개, cityscapes 는 19개로 해줫는데, 모두 unlabeled 는 빼고 실제로 의미잇는 클래스들 갯수만 센거임.
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled_Label'      ,  0 ,        0 , 'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
#     Label(  'sinus'                ,  1 ,        1 , 'sinusJ'          , 1       , False        , False        , (  0,  0,255) ),
#     Label(  'maxilla'              ,  2 ,        2 , 'boneJ'           , 2       , False        , False        , (162,156,255) ),
#     Label(  'mandible'             ,  3 ,        3 , 'boneJ'           , 2       , False        , False        , (185,181,247) ),
#     Label(  'canal'                ,  4 ,        4 , 'canalJ'          , 3       , False        , False        , ( 76, 68,212) ),
#     Label(  't_normal'             ,  5 ,        5 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  6 ,        6 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
# ]
#
#####################################################################################################################################################

# i.21.3.17.21:09) 
#    바로위에 내가 코멘트적은대로, sinus 랑 canal 의 hasInstances 를 True 로 해줘보고, maxilla 도 hasInstances 를 True 로 해줘보려함. 
#  이제 mandible 만 hasInstances False 임. maxilla 랑 mandible 의 hasIntances 에 따른 **차이좀 비교**해보려고 다르게 해줬음.
#  
#    그리고, 위에적은대로, unlabeled_Label 자체를 걍 없애버리고 총 클래스수를 8개가 아닌 7개로 바꿔주려함.
#  그리고 Label들 순서 바꿔줫고, maxilla 색깔 붉은주황색계열로 바꿔줘봄.
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'mandible'             ,  0 ,        0 , 'boneJ'           , 0       , False        , False        , (185,181,247) ),
#     Label(  'maxilla'              ,  1 ,        1 , 'boneJ'           , 0       , True         , False        , (255, 85, 79) ),
#     Label(  'sinus'                ,  2 ,        2 , 'sinusJ'          , 1       , True         , False        , (  0,  0,255) ),
#     Label(  'canal'                ,  3 ,        3 , 'canalJ'          , 2       , True         , False        , ( 76, 68,212) ),
#     Label(  't_normal'             ,  4 ,        4 , 'toothJ'          , 3       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  5 ,        5 , 'toothJ'          , 3       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  6 ,        6 , 'toothJ'          , 3       , True         , False        , (116,255, 56) ),
# ]





######## i.21.3.20.18:26) 이게 Det2 의 기존코드들대로 해준, 프레딕션 선택지에서 백그라운드는 제외한 버전. 백그라운드가 거의다 sinus로 프레딕션됐었지. #############
#         이밸류에이션시에는 이런식으로 해주자. 다만, 여기서 좀 수정할부분 있는데,
#         hasInstances 는 좀 바꿔주는게 좋을듯. [레퍼런스버전1] 처럼 치아나 임플 말고는 stuff 로. 그게 결과가 더 나은듯.
#
# i.21.3.18.9:18) 바로위처럼 mandible 의 id를 0으로 해주니, ~~instanceIds.png 에 mandible 의 id값이 0으로 기록되는데,
#  J_createPanopticImgs.py 에서 ~~instanceIds.png 로부터 coco어노png 만들어줄때 백그라운드 픽셀들의 값을 [0,0,0]으로 해주는데
#  mandible 의 id 값이 0이라서 얘도 픽셀들 값이 [0,0,0] 으로 변환돼버림. 그래서 mandible이랑 백그라운드 둘다 [0,0,0]이돼서
#  트레이닝 결과 보면 mandible과 백그라운드 전부 다 mandible 로 프레딕션하게된거임.
#    -> 즉, 의미있는 클래스들은 id값이 0이 아니어야함(지금상태의 코드들을 이용해준다면). 
#    -> 그래서 그냥 unlabeled_Label 을 다시사용해주되,
#       Det2 mvdelt깃헙버전의 J_cityscapes_panoptic.py(데이터셋레지스터해주는파일) 에서 CITYSCAPES_CATEGORIES_J 에서는 unlabeled_Label 을 없애기로.
#       (안사용하고 걍 mandible부터 id 1, trainId 0부터 시작하게해도 되긴 하겠지만)
#  참고로 Det2 의 cityscapes_panoptic.py 에서는 의미있는 클래스들의 목록만 사용하고, 
#  얘네들의 trainId 들은 0부터 연속적으로(0,1,2,...)돼있기때문에 trainId 를 "contiguous id" 로 사용해줌. 
#  즉, Det2 형식에 넣어주는 각 segment_info 의 'category_id' 값은 trainId 이고 요게 연속적으로 0,1,2,.. 일케되는거임.
#  Det2 에선 바로 이 연속적인 0,1,2,... 값들을 카테고리id 로서 사용하고,
#  Det2 형식의 각 segment_info 에 'category_id' 뿐 아니라 'id' 도 있는데, 바로 이 'id' 값이 ~~instanceIds.png 및 coco어노png 에 저장된 각 픽셀의 id값임.
#  (~~instanceIds.png 에는 10진법으로 id가 저장되고, coco어노png 에는 256진법으로 RGB로 id가 저장되지.)
#  이 id 값들은 뭐가되든 상관없고 걍 인스턴스들마다 값이 달라서 구분만 되면 되는듯. 
#  Det2 에서는, Det2 형식의 각 segment_info 의 'category_id' 와 'id' 정보를 통해서, coco어노png 의 id 값들로부터 카테고리id 값을 알아낼 수 있겠지.
#  (참고로 segments_info 에 여러개의 segment_info 가 들어잇는거지. s 있냐없냐 잘봐라.)
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled_Label'      ,  0 ,      255 , 'voidJ'           , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'mandible'             ,  1 ,        0 , 'boneJ'           , 1       , False        , False        , (185,181,247) ),
#     Label(  'maxilla'              ,  2 ,        1 , 'boneJ'           , 1       , True         , False        , (255, 85, 79) ),
#     Label(  'sinus'                ,  3 ,        2 , 'sinusJ'          , 2       , True         , False        , (  0,  0,255) ),
#     Label(  'canal'                ,  4 ,        3 , 'canalJ'          , 3       , True         , False        , ( 76, 68,212) ),
#     Label(  't_normal'             ,  5 ,        4 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
#     Label(  't_tx'                 ,  6 ,        5 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
#     Label(  'impl'                 ,  7 ,        6 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
# ]
###########################################################################################################################################################


############### i.21.3.20.18:31) 시각화용(프레딕션 선택지에 백그라운드 포함) 버전. 머 결국 [레퍼런스버전1] 이랑 똑같이 해주게될듯.################################
# i.21.3.18.20:44) 바로위처럼 하고 Det2 의 J_cityscapes_panoptic.py 에서 unlabeled_Label 을 없애줬더니, 
#  프레딕션 되지 않아야할 백그라운드가 foreground 카테고리들로 프레딕션되네;; 특히 sinus 가 많네. 아무래도 sinus가 시커머니까
#  파노영상의 어두운 백그라운드들이 다 sinus 로 프레딕션되는듯함.
#    ->그러면 unlabeled 카테고리를 따로 지정하지 않고 어케하지? threshold 를 지정해줘야하나?? 지금 이거 bowen깃헙에 질문올려본상태임.
#  암튼 그래서 다시 예전처럼 unlabeled_Label 을 살려주려함. unlabeled_Label 의 ignoreInEval 도 다시 False 로 바꿔주고.
#  아직 수정 다 못했음. 내일 해야함. 
#
# i.21.3.20.17:47) 바로위처럼 햇던게 Det2 의 panoptic seg 관련 데이터셋레지스터하는 코드들 해주는대로 따라한건데,
#  (coco 및 cityscapes 데이터셋들 레지스터하는 코드들)
#  그러면 그 코드들은 백그라운드 프레딕션을 안하는건가 햇는데, bowen깃헙에서 bowen의 답변 들으니 무슨상황인지 좀 알겟네. 
#  cityscapse 이밸류에이션 방식에선, 백그라운드는 이밸류에이션에서 제외임!
#  그니까 백그라운드도 그냥 뭐 죄다 foreground 카테고리로 프레딕션하게 하는게 더 평가에서 유리하겠지(어차피 백그라운드면 점수안깎이니까).
#  그래서 Det2 의 기존 코드들이 그런식으로(백그라운드는 프레딕션 선택지에서 없도록) 해줬던건가봄.
#    실제로 coco panoptic api 및 cityscapes 의 panoptic seg 이밸류에이션하는 코드들 (둘다 알렉스 뭐시기가 만든건지 거의 같음)
#  을 조사해보니, coco 나 cityscapes 나 모두 gt 백그라운드(VOID)는 모델이 뭘로 프레딕션하든지 평가에 반영 안됨. 
#  뿐만아니라, 코드보니까, gt 가 iscrowd=1(ex: 사람이나 차 등이 잔뜩모여있는경우) 인 경우에도 마찬가지로 이밸류에이션에 반영 안함.
#  다시간단히말하면, 이미지에서 gt 가 VOID(백그라운드)거나 iscrowd=1 인 부분들은 평가에 반영 안됨. COCO나 cityscapes 나 마찬가지.
#    따라서, 요 코드로 이밸류에이션할땐 프레딕션할 선택지에서 백그라운드(VOID. 즉 'unlabeled')를 없애주는게 조금이라도 더 유리할거고,
#  그냥 시각화해서 사람눈으로 볼 용도로는 프레딕션할 선택지에 백그라운드도 포함시켜주는게 좋겠네.
#  (안그러면 바로위처럼 해줬을때처럼 sinus 등 foreground 카테고리들이 백그라운드를 뒤덮을테니)
#    자, 그럼, 
#  (1) 일단 시각화할 용도로 백그라운드(내플젝의경우 'unlabeled_Label') 도 프레딕션하게 해주자. 
#      바로아래 labels 및 관련 코드들을 그렇게 설정해줄거임.
#  (2) (1)이 잘 되고나면, 이밸류에이션 용도로 프레딕션 선택지에서 백그라운드 제외해서도 하자.
#      지금 코멘트아웃돼있는 바로 위 labels 이용하고 관련 코드들도 그에맞게 변경해주면 되지.
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
###########################################################################################################################################################


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
print(f'j) name2label has been made!!!!!!')
# print(f'j) name2label: {name2label}')

# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) } # inverse mapping(trainId->Label 맵핑)에선 위 Label리스트의 (동일한 trainId를 가지는 Label들 중)1번째녀석을 사용하기위해 reverse 해줌. 안그러면 1번째가 아니라 마지막놈이 사용될테니./i.21.3.5.19:00.
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
