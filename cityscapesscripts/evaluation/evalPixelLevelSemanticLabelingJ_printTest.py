


from collections import namedtuple
import sys
import numpy as np

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


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color(RGB)
    Label(  'unlabeled_Label'      ,  0 ,        0,  'voidJ'           , 0       , False        , False        , (  0,  0,  0) ),
    Label(  'mandible'             ,  1 ,        1 , 'boneJ'           , 1       , False        , False        , (135,128,255) ),
    Label(  'maxilla'              ,  2 ,        2 , 'boneJ'           , 1       , False        , False        , (207,221,255) ),
    Label(  'sinus'                ,  3 ,        3 , 'sinusJ'          , 2       , False        , False        , (  0,  0,255) ),
    Label(  'canal'                ,  4 ,        4 , 'canalJ'          , 3       , False        , False        , (255,  0,  0) ),  # i. canal 이 젤 고난이도니까, 나중에 hasInstances True로도 실험해보자 어찌되는지.
    Label(  't_normal'             ,  5 ,        5 , 'toothJ'          , 4       , True         , False        , ( 66,158, 27) ),
    Label(  't_tx'                 ,  6 ,        6 , 'toothJ'          , 4       , True         , False        , ( 88,214, 34) ),
    Label(  'impl'                 ,  7 ,        7 , 'toothJ'          , 4       , True         , False        , (116,255, 56) ),
]


id2label = {label.id:label for label in labels}


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()


# Remaining params
args.evalInstLevelScore = True
args.evalPixelAccuracy  = False
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
# args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
# args.bold               = colors.BOLD if args.colorized else ""
# args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False



# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(args):
    args.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.id)
    maxId = max(args.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)




# Print confusion matrix
#
# i.21.4.14.19:33) 아래 내가 붙여놓은 표와 같이 컨퓨젼매트릭스 그려지는듯.
#  (어떤이유인지 코랩에서는 표로 안그려지고 한줄로 쭉 붙어서 출력되지만, 적당히 줄바꿈 해주면 아래 표처럼 됨.) 
#    근데, 현재 args.normalized=True 로 돼있기때문에 
#  getMatrixFieldValue 함수
#  (args.normalized 값에따라, 컨퓨전매트릭스의 각 셀의 값(픽셀갯수)을 그대로 리턴하거나, row총합으로 나눈 노말라이즈된값을 리턴)
#  의 리턴값이 컨퓨젼매트릭스의 각 셀의 값 그대로가 아니고, 해당 셀이 속한 row 의 모든셀의값들 총합으로 나눈 값이 리턴됨.
#  그렇게 노말라이즈된 결과가 바로아래 내가 붙여놓은 테이블임.
#    그리고 Prior 는 뭐냐면, (노말라이즈시키기전의, 즉 아래표와는 다른, 원래의)컨퓨젼매트릭스에서
#  각 row 의 모든픽셀수 총합을 컨퓨젼매트릭스 전체 픽셀수 총합으로 나눈 값임 (각 row 마다 prior 값이 계산되는거지).
#  즉, 각 row 는 gt(ground truth)를 의미하니까,
#  전체 픽셀수(모든 테스트이미지들의 픽셀수 총 합) 중에서 각각의 gt 클래스가 점유하는 픽셀갯수를 비율로 나타낸것임. 
#  getPrior 함수 보면 여기서 말하는 prior 가 뭔지 아주 간단히 알수있음. 
#  -------------- ------ ------ ------ ------ ------ ------ ------ ------ ------- 
#                |  u   |  m   |  m   |  s   |  c   |  t   |  t   |  i   | Prior |
#  -------------- ------ ------ ------ ------ ------ ------ ------ ------ ------- 
#  unlabeled_Lab | 0.98   0.01   0.00   0.01   0.00   0.00   0.00   0.00  0.5384      
#       mandible | 0.04   0.94   0.00   0.00   0.01   0.00   0.00   0.00  0.2092       
#        maxilla | 0.01   0.00   0.92   0.01   0.00   0.02   0.03   0.01  0.0433
#          sinus | 0.03   0.00   0.03   0.94   0.00   0.00   0.00   0.00  0.0795  
#          canal | 0.00   0.25   0.00   0.00   0.75   0.00   0.00   0.01  0.0173   
#       t_normal | 0.03   0.02   0.01   0.00   0.00   0.83   0.11   0.00  0.0282       
#           t_tx | 0.02   0.04   0.01   0.00   0.00   0.06   0.87   0.00  0.0403        
#           impl | 0.03   0.01   0.00   0.00   0.00   0.00   0.16   0.81  0.0439 
#  -------------- ------ ------ ------ ------ ------ ------ ------ ------ -------  
#
# i.21.4.14.20:46) 참고로, cityscapesScripts 깃헙의 Issues 에서 mcordts (cityscapesScripts 주 개발자) 가 남긴 답변 일부 발췌 (prior 로 검색해서 나온 내용):
#  "...The script outputs a confusion matrix first. The letters on top are the first letters of the label names in the left column. 
#  The prior is the proportion of pixels with the respective ground truth label. The nIoU is the iIoU of the paper. 
#  Apparently, I used an n for normalized there instead of an i for instance."
#    ->prior 가 뭔지 역시 내생각대로임. nIoU 의 n 이 노말라이즈의 의미인가했던것도 내추측이 맞았네. nIoU 가 iIou 나 똑같은거임. 
# 
def printConfMatrix(confMatrix, args):
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" ")) # i. 코랩에서는 여기서 줄바꿈이 안되고있음. end 값을 정해주지 않았으니 기본값인 \n 이 적용되어야할텐데 왜안되지? /21.4.14.20:54.

    # print label names
    print("\b{text:>{width}} |".format(width=13, text=""), end=' ')
    for label in args.evalLabels:
        print("\b{text:^{width}} |".format(width=args.printRow, text=id2label[label].name[0]), end=' ')
    print("\b{text:>{width}} |".format(width=6, text="Prior")) # i. 여기서도 마찬가지로 줄바꿈 안되고있고. /21.4.14.20:55.

    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" ")) # i. 여기서도 마찬가지로 줄바꿈 안되고있고. 아래 코드들 전부 다 그러함. /21.4.14.20:55.

    # # print matrix
    # for x in range(0, confMatrix.shape[0]):
    #     if (not x in args.evalLabels):
    #         continue
    #     # get prior of this label
    #     prior = getPrior(x, confMatrix)
    #     # skip if label does not exist in ground truth
    #     if prior < 1e-9:
    #         continue

    #     # print name
    #     name = id2label[x].name
    #     if len(name) > 13:
    #         name = name[:13]
    #     print("\b{text:>{width}} |".format(width=13,text=name), end=' ')
    #     # print matrix content
    #     for y in range(0, len(confMatrix[x])):
    #         if (not y in args.evalLabels):
    #             continue
    #         matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, args) # i. 요놈이 각 셀의 값을 내뱉는놈. args.normalized=True/False 에 따라 리턴값 달라짐. /21.4.14.20:01. 
    #         print(getColorEntry(matrixFieldValue, args) + "\b{text:>{width}.2f}  ".format(width=args.printRow, text=matrixFieldValue) + args.nocol, end=' ')
    #     # print prior
    #     print(getColorEntry(prior, args) + "\b{text:>{width}.4f} ".format(width=6, text=prior) + args.nocol)
    
    # # print line
    # print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    # for label in args.evalLabels:
    #     print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    # print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end=' ')



    
cmatJ=generateMatrix(args)

printConfMatrix(cmatJ, args) 
