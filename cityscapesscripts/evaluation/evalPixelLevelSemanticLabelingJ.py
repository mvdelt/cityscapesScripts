#!/usr/bin/python
#
#
##################################################################################################################### 
# i.21.3.29.12:42) 이 파일 설명:
#  evalPixelLevelSemanticLabeling.py 를 
#  내플젝(panoptic deeplab 이용해서 치과파노라마 panoptic segmentation) 에 맞게 수정해주기위해 
#  전날(3월28일)에 만든 파일. 
#  evalPixelLevelSemanticLabeling.py 에 계속 코멘트달고 이것저것 수정하다가, 
#  cityscapes 플젝 돌릴때랑 달라지는부분 있어서 
#  그냥 그파일 그대로 복사한뒤에 내플젝에맞게 수정해주려고 만들었음. 
#####################################################################################################################
#
# 
# The evaluation script for pixel-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# Note that the script is a faster, if you enable cython support.
# WARNING: Cython only tested for Ubuntu 64bit OS.
# To enable cython, run
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# To run this script, make sure that your results are images,
# where pixels encode the class IDs as defined in labels.py.
# Note that the regular ID is used, not the train ID.
# Further note that many classes are ignored from evaluation.
# Thus, authors are not expected to predict these classes and all
# pixels with a ground truth label that is ignored are ignored in
# evaluation.

# python imports
from __future__ import print_function, absolute_import, division
import os, sys
import platform
import fnmatch

try:
    from itertools import izip
except ImportError:
    izip = zip

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import *

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        from cityscapesscripts.evaluation import addToConfusionMatrix
    except:
        CSUPPORT = False


###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
#
# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable CITYSCAPES_RESULTS
#   - environment variable CITYSCAPES_DATASET/results
#   - ../../results/"
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
# <city>_123456_123456*.png
# for a ground truth filename
# <city>_123456_123456_gtFine_labelIds.png
def getPrediction( args, groundTruthFile ):  
    
    # i. groundTruthFile ex: ~~/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png 
    #  즉, 지금 이 getPrediction 함수는 *1개의* gt png파일에 대해 작동하는거임. /21.3.28.11:15.

    # determine the prediction path, if the method is first called
    if not args.predictionPath: # i. Det2 의 cityscapes_evaluation.py 에선 args.predictionPath 를 임시폴더(의경로)로 지정해주기때문에 이 if 실행안됨. /21.3.28.9:04.
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        elif 'CITYSCAPES_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['CITYSCAPES_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        args.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk: # i. Det2 의 cityscapes_evaluation.py 에선 args.predictionWalk 를 None 으로 해주므로, 이 if 실행됨. /21.3.28.9:07. 
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk
        print(f'j) 예상 args.predictionWalk: [("/임시/폴더의/경로", [~~_pred.png, ~~_pred.png, ...])] 이렇게 튜플하나만있을거임 내예상엔.')
        print(f'j) 실제 args.predictionWalk: {walk}') # [('/tmp/cityscapes_eval_se7hxspp', ['munster_000150_000019_leftImg8bit_pred.png', 'frankfurt_000001_014221_leftImg8bit_pred.png', ...])]
        print(f'j) (예상값 1) len(args.predictionWalk): {len(walk)}') # 1 
        print(f'j) (예상값 500) len(args.predictionWalk[0][1]): {len(walk[0][1])}') # 500 
        # i. 바로위 print 출력결과들 전부 내 예상대로임. /21.3.28.11:29. 

    # i. groundTruthFile: ~~_gtFine_labelIds.png (내플젝말고 cityscapes플젝의경우.)/21.3.28.9:44. 
    #  ex: ~~/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png /21.3.28.9:48. 
    #
    # i.21.3.28.10:54) getCsFileInfo 의 리턴값의 정체는 namedtuple 임. 
    #  CsFile = namedtuple('csFile', ['city', 'sequenceNb', 'frameNb', 'type', 'type2', 'ext']) 
    #  csFile = CsFile(*['frankfurt', '000000', '000294', 'gtFine', 'labelIds', 'png'])
    csFile = getCsFileInfo(groundTruthFile)
    filePattern = "{}_{}_{}*.png".format( csFile.city , csFile.sequenceNb , csFile.frameNb ) # i. ex: "frankfurt_000000_000294*.png" 

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile # i. ex: '/임시/폴더의/경로/frankfurt_000000_000294_leftImg8bit_pred.png' 


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()

# Where to look for Cityscapes
if 'CITYSCAPES_DATASET' in os.environ:
    args.cityscapesPath = os.environ['CITYSCAPES_DATASET']
else:
    # i.21.3.30.15:54) 지금보니 코랩의 내플젝의경우 요거 적용됐겠네.
    # (코랩에서 cityscapes 데이터셋 돌려줄때는 환경변수 'CITYSCAPES_DATASET' 설정했었는데, 
    #  내플젝돌릴땐 안했지. 트레이닝시킬 데이터 준비하는 파일들에서 하드코딩으로 경로들 지정해줬었지.) 
    #  난 트레이닝시킬때 준비해주는 파일들에 내플젝에맞게 경로 지정해줘서
    #  환경변수 'CITYSCAPES_DATASET' 이거 셋팅 안해줘도 되는줄 알았는데(트레이닝 및 시각화결과출력 까지는 그래도됐지),
    #  이제 이밸류에이션 파일들 돌려주려면 경로 또 설정 해줘야겠네. 
    #  환경변수 'CITYSCAPES_DATASET' 이용해서 지정해주든지, 
    #  각 파일들에 하드코딩시키든지(예를들어 바로아랫줄코드를 바꾸는 식으로) 어떻게든. 
    #  근데 지금 이 args.cityscapesPath 는 결과 저장용인듯하니 당장 안바꿔줘도 작동에 문제는 없겠네. 
    args.cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

if 'CITYSCAPES_EXPORT_DIR' in os.environ:
    export_dir = os.environ['CITYSCAPES_EXPORT_DIR']
    if not os.path.isdir(export_dir):
        raise ValueError("CITYSCAPES_EXPORT_DIR {} is not a directory".format(export_dir))
    args.exportFile = "{}/resultPixelLevelSemanticLabeling.json".format(export_dir)
else:
    args.exportFile = os.path.join(args.cityscapesPath, "evaluationResults", "resultPixelLevelSemanticLabeling.json")
# Parameters that should be modified by user
args.groundTruthSearch  = os.path.join( args.cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_labelIds.png" )
# i.21.3.30.15:45) ->args.groundTruthSearch 는 이 파일의 main() 메서드에서만 사용함.
#  지금 Det2 의 cityscapes_evaluation.py 에서는 이 파일의 evaluateImgLists 를 사용하고있기때문에 
#  args.groundTruthSearch 값을 내플젝에맞게 안바꿔줘도 잘 작동하는것임. 
#  (Det2 에서 이 파일의 evaluateImgLists 사용해주기 전에 gt 파일들을 이미 찾아서 넣어주고있지. 난 그부분을 내플젝에맞게 바꿔준거고.) 

# Remaining params
args.evalInstLevelScore = True
args.evalPixelAccuracy  = False
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ""
args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False



# args.avgClassSize       = {
#     "bicycle"    :  4672.3249222261 ,
#     "caravan"    : 36771.8241758242 ,
#     "motorcycle" :  6298.7200839748 ,
#     "rider"      :  3930.4788056518 ,
#     "bus"        : 35732.1511111111 ,
#     "train"      : 67583.7075812274 ,
#     "car"        : 12794.0202738185 ,
#     "person"     :  3462.4756337644 ,
#     "truck"      : 27855.1264367816 ,
#     "trailer"    : 16926.9763313609 ,
# }
# i.21.3.28.21:44) 
#  TODO 내플젝에선 args.avgClassSize 를 위의 cityscapes 플젝에 맞춰놓은값 말고 아래의 내플젝에 맞는 값 써야함.
#  이 값들의 정체가 뭐지?? 아마도 area 를 평균낸것같은데, 일단 계산할시간없으니 대충 써놔봄.
#   -> i. 걍 직관적으로 내가 생각하는게 맞는듯함. 
#      코드 살펴보니, 아마도, 각 thing 클래스의 모든 gt 인스턴스 area 들을 평균낸 값인듯.(걍 내가생각하는대로임) /21.3.29.0:05. 
#   -> i. 아닌듯!! gt "인스턴스" area 평균이 아니고, gt "클래스"(인스턴스들의 뭉탱이)의 area 평균인듯. 말그대로 avg"ClassSize" 임. /21.3.30.0:26.
#  TODO 이 값들의 정체가 뭔지 확인해서, 계산 제대로 해서 이밸류에이션 다시 돌려줄것. 
#  TODO 기존 cityscapes 데이터로 이밸류에이션 돌려보면 IoU 값들보다 nIoU 값들이 조금씩 더 작은데,
#   지금 요 avgClassSize 대충임시로정한값으로 내플젝 돌려보니 nIoU 값이 드뎌 0이아닌 어떤 값들이 나오긴 하는데, IoU 값들보다 조금씩 더 크다. 
#   다시 제대로된 값으로 해준뒤에 IoU 랑 nIoU 크기비교 해보고, 항상 IoU > nIoU 인지 생각해보자. 
#  vTODO 지금보니 gt json파일에는 카테고리가 7개뿐이네?!!! unlabeled 카테고리는 없잖아 생각해보니!!!! 이거 다시 생각해봐라!!!!!!!!! 
#   -> i. ㅡㅡ;; 먼소리냐 어차피 gt png파일들(~~instanceIds.png 등의) 생성해줄때 기본적으로 백그라운드값 설정한뒤에 그려주잖아;;; 
#      coco-annotator 에서 다운받은 "시초" 어노json파일에서 백그라운드에대한 어노테이션정보가 없더라도 gt png파일들 생성시에는 다 생기는거지. 
#      백그라운드 픽셀값 셋팅한뒤에 각 폴리곤들을 그려주니까. /21.3.29.0:01. 
#
# i.21.3.29.0:02) thing 클래스들에 대한 정보만 넣어주면 됨. 
#  지금현재는 내플젝에서 thing클래스들이 요 세개니까 얘네들의 정보만 넣어주면 됨.
# args.avgClassSize       = {
#     # "unlabeled_Label"    :  1000000 ,
#     # "mandible"    : 350000 ,
#     # "maxilla" :  80000 ,
#     # "sinus"      :  150000 ,
#     # "canal"        : 30000 ,
#     "t_normal"      : 50000 ,
#     "t_tx"        : 50000 ,
#     "impl"     :  50000 ,
# }
#
# i.21.4.1.1:53) 내플젝에맞게 계산해준 avgClassSize 를 임포트. 
#  (바로위처럼 하드코딩으로 avgClassSize 를 직접 작성해주는것이 아니고. 일케하면 새롭게 데이터(어노테이션해준거) 추가해도 걍 자동으로 계산되지.) 
#  참고로 val 데이터셋 파노 2개에 대해 현재 계산해준 결과: 
#  {'t_normal': 9904.545454545454, 't_tx': 17264.0, 'impl': 24174.428571428572} 
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabelingJ_calculateAvgClassSizeJ import avgClassSizeJ, avgClassSizeJ_2
print(f'j) avgClassSizeJ    : {avgClassSizeJ}')
print(f'j) avgClassSizeJ_2  : {avgClassSizeJ_2}')
args.avgClassSize = avgClassSizeJ_2 
print(f'j) args.avgClassSize: {args.avgClassSize}') 

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


#########################
# Methods
#########################


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

def generateInstanceStats(args):
    instanceStats = {}
    instanceStats["classes"   ] = {}
    instanceStats["categories"] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats["classes"][label.name] = {}
            instanceStats["classes"][label.name]["tp"] = 0.0
            instanceStats["classes"][label.name]["tpWeighted"] = 0.0
            instanceStats["classes"][label.name]["fn"] = 0.0
            instanceStats["classes"][label.name]["fnWeighted"] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats["categories"][category] = {}
        instanceStats["categories"][category]["tp"] = 0.0
        instanceStats["categories"][category]["tpWeighted"] = 0.0
        instanceStats["categories"][category]["fn"] = 0.0
        instanceStats["categories"][category]["fnWeighted"] = 0.0
        instanceStats["categories"][category]["labelIds"] = labelIds

    return instanceStats


# Get absolute or normalized value from field in confusion matrix.
def getMatrixFieldValue(confMatrix, i, j, args):
    # i.21.4.14.19:49) 기본값은 저위에서 args.normalized=True 로 돼있는데,
    #  지금 이 파이썬파일(지금 이 파이썬모듈, 즉 evalPixelLevelSemanticLabelingJ.py)
    #  을 사용해주는곳에서 args.normalized 값을 따로 지정해주지 않으면 이 기본값이 적용되지. 
    #  지금 Det2 의 cityscapes_evaluation.py 에서 지금 이 파이썬파일(파이썬모듈)을 사용해주는데, 
    #  args 값들 설정해줄때 args.normalized 는 따로 설정 안해줫기때매 현재 이 파일의 저위에서 기본값으로 정해둔 args.normalized=True 상태임.     
    if args.normalized: 
        rowSum = confMatrix[i].sum()
        if (rowSum == 0):
            return float('nan')
        return float(confMatrix[i][j]) / rowSum
    else:
        return confMatrix[i][j]

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    labelName = id2label[label].name
    if not labelName in instStats["classes"]:
        return float('nan')

    tp = instStats["classes"][labelName]["tpWeighted"]
    fn = instStats["classes"][labelName]["fnWeighted"]
    # false postives computed as above
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate prior for a particular class id.
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, args):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, args):
    # All labels in this category
    labels = category2labels[category]
    # The IDs of all valid labels in this category
    labelIds = [label.id for label in labels if not label.ignoreInEval and label.id in args.evalLabels]
    # If there are no valid labels, then return NaN
    if not labelIds:
        return float('nan')

    # the number of true positive pixels for this category
    # this is the sum of all entries in the confusion matrix
    # where row and column belong to a label ID of this category
    tp = np.longlong(confMatrix[labelIds,:][:,labelIds].sum())

    # the number of false negative pixels for this category
    # that is the sum of all rows of labels within this category
    # minus the number of true positive pixels
    fn = np.longlong(confMatrix[labelIds,:].sum()) - tp

    # the number of false positive pixels for this category
    # we count the column sum of all labels within this category
    # while skipping the rows of ignored labels and of labels within this category
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, args):
    if not category in instStats["categories"]:
        return float('nan')
    labelIds = instStats["categories"][category]["labelIds"]

    tp = instStats["categories"][category]["tpWeighted"]
    fn = instStats["categories"][category]["fnWeighted"]

    # the number of false positive pixels for this category
    # same as above
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom


# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, args ):
    # write JSON result file
    wholeData = {}
    wholeData["confMatrix"] = confMatrix.tolist()
    wholeData["priors"] = {}
    wholeData["labels"] = {}
    for label in args.evalLabels:
        wholeData["priors"][id2label[label].name] = getPrior(label, confMatrix)
        wholeData["labels"][id2label[label].name] = label
    wholeData["classScores"] = classScores
    wholeData["classInstScores"] = classInstScores
    wholeData["categoryScores"] = categoryScores
    wholeData["categoryInstScores"] = categoryInstScores
    wholeData["averageScoreClasses"] = getScoreAverage(classScores, args)
    wholeData["averageScoreInstClasses"] = getScoreAverage(classInstScores, args)
    wholeData["averageScoreCategories"] = getScoreAverage(categoryScores, args)
    wholeData["averageScoreInstCategories"] = getScoreAverage(categoryInstScores, args)

    if perImageStats:
        wholeData["perImageScores"] = perImageStats

    return wholeData

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportFile)


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
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end='\n') # i. 코랩에서는 여기서 줄바꿈이 안되고있음. end 값을 정해주지 않았으니 기본값인 \n 이 적용되어야할텐데 왜안되지? /21.4.14.20:54.
    print() # i. <-코랩에서 출력시 바로윗줄프린트에서 줄바꿈이 안돼서 내가 집어넣음. 걍 이렇게해주니 그제서야 줄바꿈 하네;; 왜 코랩에서만 안되지? 뭐암튼 이렇게해서 일단은 해결. /21.4.24.19:12.

    # print label names
    print("\b{text:>{width}} |".format(width=13, text=""), end=' ')
    for label in args.evalLabels:
        print("\b{text:^{width}} |".format(width=args.printRow, text=id2label[label].name[0]), end=' ')
    print("\b{text:>{width}} |".format(width=6, text="Prior"), end='\n') # i. 여기서도 마찬가지로 줄바꿈 안되고있고. /21.4.14.20:55.
    print() # i. <-코랩에서 출력시 바로윗줄프린트에서 줄바꿈이 안돼서 내가 집어넣음. /21.4.24.19:13.

    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end='\n') # i. 여기서도마찬가지. 아래 코드들 전부 다 그러함. end='\n' 이라고 명시적으로 적어줘도 안됨. 코랩에서만 그러는듯./21.4.14.20:55.
    print() # i. <-코랩에서 출력시 바로윗줄프린트에서 줄바꿈이 안돼서 내가 집어넣음. /21.4.24.19:13.

    # print matrix
    for x in range(0, confMatrix.shape[0]):
        if (not x in args.evalLabels):
            continue
        # get prior of this label
        prior = getPrior(x, confMatrix)
        # skip if label does not exist in ground truth
        if prior < 1e-9:
            continue

        # print name
        name = id2label[x].name
        if len(name) > 13:
            name = name[:13]
        print("\b{text:>{width}} |".format(width=13,text=name), end=' ')
        # print matrix content
        for y in range(0, len(confMatrix[x])):
            if (not y in args.evalLabels):
                continue
            matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, args) # i. 요놈이 각 셀의 값을 내뱉는놈. args.normalized=True/False 에 따라 리턴값 달라짐. /21.4.14.20:01. 
            print(getColorEntry(matrixFieldValue, args) + "\b{text:>{width}.2f}  ".format(width=args.printRow, text=matrixFieldValue) + args.nocol, end=' ')
        # print prior
        print(getColorEntry(prior, args) + "\b{text:>{width}.4f} ".format(width=6, text=prior) + args.nocol)
        print() # i. <-코랩에서 출력시 바로윗줄프린트에서 줄바꿈이 안돼서 내가 집어넣음. /21.4.24.19:13. 
    
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    # print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "))
    print() # i. <-코랩에서 출력시 바로윗줄프린트에서 줄바꿈이 안돼서 내가 집어넣음.(바로위 프린트는 원소스코드에선 end=' '지만, 줄바꿈 안해주면 그다음 프린트가 바로이어서 출력돼버림.))/21.4.24.19:13. 



# i.21.4.24.20:34) confusion matrix 를 plot 도 해보려고 내가 만들어줌. 
def plotConfMatrixJ(confMatrix):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # array = [[13,1,1,0,2,0],
    #         [3,9,6,0,1,0],
    #         [0,0,16,2,0,0],
    #         [0,0,0,13,0,0],
    #         [0,0,0,0,15,0],
    #         [0,0,1,0,0,15]]

    
    # for label in args.evalLabels:
    #     print("\b{text:^{width}} |".format(width=args.printRow, text=id2label[label].name[0]), end=' ')
    # print("\b{text:>{width}} |".format(width=6, text="Prior"), end='\n') # i. 여기서도 마찬가지로 줄바꿈 안되고있고. /21.4.14.20:55.


    # df_cm = pd.DataFrame(confMatrix, index = [i for i in "ABCDEF"],columns = [i for i in "abcdef"])
    df_cm = pd.DataFrame(confMatrix, index = [i for i in "ummsctti"], columns = [i for i in "ummsctti"])

    # plt.figure(figsize=(10,17))
    sns.set(font_scale=1.4) # for label size
    
    # sns.heatmap(df_cm, annot=True,  cmap=sns.cm.rocket_r, annot_kws={"size": 16}) # font size
    # sns.heatmap(df_cm, annot=True,  cmap="Blues", annot_kws={"size": 16}) # font size
    sns.heatmap(df_cm, annot=True,  cmap="YlGnBu", annot_kws={"size": 16}) # font size
    # sns.heatmap(df_cm, annot=True,  cmap="twilight", annot_kws={"size": 16}) # font size

    # i. 코랩에선 코랩 셀에서 바로 이거 실행하면 플롯 출력되지만, 
    #  파일로 실행시키면 플롯 출력 안됨. 그래서 걍 코랩클라우드컴에 저장해주려함.
    # plt.show() 

    # plt.savefig('/content/confMatrixJ.png', bbox_inches='tight', dpi=1200)
    plt.savefig('/content/confMatrixJ.png', dpi=1200)








# Print intersection-over-union scores for all classes.
def printClassScores(scoreList, instScoreList, args):
    if (args.quiet):
        return
    print(args.bold + "classes          IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for label in args.evalLabels:
        if (id2label[label].ignoreInEval):
            continue
        labelName = str(id2label[label].name)
        iouStr = getColorEntry(scoreList[labelName], args) + "{val:>5.3f}".format(val=scoreList[labelName]) + args.nocol
        niouStr = getColorEntry(instScoreList[labelName], args) + "{val:>5.3f}".format(val=instScoreList[labelName]) + args.nocol
        print("{:<14}: ".format(labelName) + iouStr + "    " + niouStr)

# Print intersection-over-union scores for all categorys.
def printCategoryScores(scoreDict, instScoreDict, args):
    if (args.quiet):
        return
    print(args.bold + "categories       IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for categoryName in scoreDict:
        if all( label.ignoreInEval for label in category2labels[categoryName] ):
            continue
        iouStr  = getColorEntry(scoreDict[categoryName], args) + "{val:>5.3f}".format(val=scoreDict[categoryName]) + args.nocol
        niouStr = getColorEntry(instScoreDict[categoryName], args) + "{val:>5.3f}".format(val=instScoreDict[categoryName]) + args.nocol
        print("{:<14}: ".format(categoryName) + iouStr + "    " + niouStr)

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args):

    # i.21.3.29.0:39) -> # i.21.3.29.22:49) 잘못알던부분 수정 (pred 나 gt 나 둘다 클래스id 들을 담고있음). 
    #  predictionImgList 는 모델이 프레딕션한 클래스id (label.trainId 말고 그냥 label.id) 들이 그려져있고 VOID는 255로 그려진 
    #  png 의 경로들의 리스트임.(모델의 인퍼런스 아웃풋에서 각 이미지에해당하는 dict 의 "sem_seg" 의 정보에 따라 그려준것임.) 
    #  groundTruthImgList 는 내플젝의경우 ~~_labelTrainIds.png 즉 클래스id(인데 train용 id (trainId)) 가 그려진 gt png 의 경로들의 리스트임. 
    #  (내플젝에선 현재는 label.trainId 랑 label.id 랑 똑같기때문에 상관없음. 
    #   만약 달랐으면, gt 로 사용할 ~~_labelIds.png 를 만들어주든지, 아니면 모델이 프레딕션한거 png 로 그려줄때 label.trainId 로 그려주면 되지.) 
    #  (참고로, 일반적으로는 클래스라는 표현이나 카테고리라는 표현이나 똑같은의미로 쓰이는데, 
    #   cityscapes 의 labels.py 에서는 'category' 가 슈퍼카테고리를 의미함. 
    #   예를들어 차,자전거 등을 모두 vehicle 이라고한다면 vehicle 이 슈퍼카테고리일건데 
    #   이걸 cityscapes 에서는 'category' 라고 한다는거지. 나중에 헷갈릴까봐 적어둠.) 

    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")
    confMatrix    = generateMatrix(args)
    instStats     = generateInstanceStats(args)
    perImageStats = {}
    nbPixels      = 0


    # i.21.4.22.17:44) 사람의 프레딕션결과도 이밸류에이션해주기로하면서, 사람/모델 누구의 프레딕션결과인지 출력좀해주려고 추가함.(현재는 사람일때만 args.modelNameJ 값 지정해줬음.)
    if hasattr(args, 'modelNameJ'):
        print(f'======={args.modelNameJ}에 대한 이밸류에이션 결과임.=======') 


    if not args.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        #print "Evaluate ", predictionImgFileName, "<>", groundTruthImgFileName
        nbPixels += evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instStats, perImageStats, args)

        # sanity check
        if confMatrix.sum() != nbPixels:
            printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

        if not args.quiet:
            print("\rImages Processed: {}".format(i+1), end=' ')
            sys.stdout.flush()
    if not args.quiet:
        print("\n")

    # sanity check
    if confMatrix.sum() != nbPixels:
        printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

    # print confusion matrix
    if (not args.quiet):
        printConfMatrix(confMatrix, args)
    
    
    # i.21.4.24.20:49) plot confusion matrix. 내가추가. 
    if (not args.quiet):
        plotConfMatrixJ(confMatrix)
    

    # Calculate IOU scores on class level from matrix
    classScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)

    # Calculate instance IOU scores on class level from matrix
    classInstScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        print("")
        printClassScores(classScoreList, classInstScoreList, args)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classInstScoreList , args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # Calculate IOU scores on category level from matrix
    categoryScoreList = {}
    for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,args)

    # Calculate instance IOU scores on category level from matrix
    categoryInstScoreList = {}
    for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        printCategoryScores(categoryScoreList, categoryInstScoreList, args)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryInstScoreList, args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, args )
    # write result file
    if args.JSONOutput:
        writeJSONFile( allResultsDict, args)

    # return confusion matrix
    return allResultsDict

# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instanceStats, perImageStats, args):
    # Loading all resources for evaluation.
    try:
        predictionImg = Image.open(predictionImgFileName)
        predictionNp  = np.array(predictionImg) ######################## i.21.3.29.23:49)
    except:
        printError("Unable to load " + predictionImgFileName)
    try:
        groundTruthImg = Image.open(groundTruthImgFileName)
        groundTruthNp = np.array(groundTruthImg) ######################## i.21.3.29.23:49)
    except:
        printError("Unable to load " + groundTruthImgFileName)
    # load ground truth instances, if needed
    if args.evalInstLevelScore:
        # groundTruthInstanceImgFileName = groundTruthImgFileName.replace("labelIds","instanceIds") 
        # i. TODO 내플젝에선 바로위의 원래코드 대신 아래코드를 사용해야함!! 파이썬의 스트링.replace 는, replace 할 스트링이 없으면 그냥 원래스트링 그대로를 리턴함.
        #  지금 groundTruthImgFileName 의 값은 내플젝의경우 path/to/~~_labelTrainIds.png 이런식일텐데, 
        #  위의 기존코드 그대로쓰면 바꿔줄 "labelIds" 스트링이 없으니 그냥 아무것도 안바꾸고 그대로 리턴한다는거지.
        #  그러면 ~~_instanceIds.png 를 열어야하는데 그게아니라 ~~_labelTrainIds.png 를 열겠지. 
        #  아마도 그래서 코랩에서 이밸류에이션돌렸을때 thing클래스들의 nIoU 값들이 죄다 0으로 나왔던게 아닌가 싶음. 이제 고쳤으니 함 다시 돌려보자. /21.3.28.21:16. 
        #   ->일케해주니 죠아래 args.avgClassSize[label.name] 에서 키에러가 뜨네 t_normal 키가 없다고. 
        #     일케되는게 정상인데, 위의 원래코드썼을땐 이런문제 안생기고 넘어갓엇음;; 걍 넘어간 이유가 잇겟지 조사해보면 나올거임 일단패스. 
        #     일단 avgClassSize 를 내플젝에 맞게 고쳐주자. /21.3.28.21:42. 
        groundTruthInstanceImgFileName = groundTruthImgFileName.replace("labelTrainIds","instanceIds") 
        try:
            instanceImg = Image.open(groundTruthInstanceImgFileName)
            instanceNp  = np.array(instanceImg) ######################## i.21.3.29.23:49)
        except:
            printError("Unable to load " + groundTruthInstanceImgFileName)

    # Check for equal image sizes
    if (predictionImg.size[0] != groundTruthImg.size[0]):
        printError("Image widths of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if (predictionImg.size[1] != groundTruthImg.size[1]):
        printError("Image heights of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if ( len(predictionNp.shape) != 2 ):
        printError("Predicted image has multiple channels.")

    imgWidth  = predictionImg.size[0]
    imgHeight = predictionImg.size[1]
    nbPixels  = imgWidth*imgHeight

    # Evaluate images
    if (CSUPPORT):
        # using cython
        confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, args.evalLabels)
    else:
        # the slower python way     
        # i.21.3.30.0:09) ->일단 요 느린 파이썬방식 코드라도 이해완료. confMatrix 의 각 칸에 픽셀갯수 넣어주는 작업 하는거임. 
        #  (참고로 컨퓨젼매트릭스는 gt 클래스들과 pred된 클래스들에 대해 각각의 경우에 픽셀갯수 넣어준 매트릭스. 클래스갯수 x 클래스갯수 만큼 칸이 있는거지.) 
        # i.21.4.11.10:08) confusion matrix 에서 TP,TN,FP,FN 이해완료. TP,TN,FP,FN 은 "각 클래스별로" 생각해야하는거네. 
        #  즉, 클래스에따라 confusion matrix 의 특정 칸이 TP 일수도있고 TN 일수도있는거임.
        #  좀더자세히는, confusion matrix 의 diagonal 칸들은 TP/TN 가능, 그 외의 칸들은 TP 말고 다(FP/FN/TN) 가능하네. 
        encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
        encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

        values, cnt = np.unique(encoded, return_counts=True)

        for value, c in zip(values, cnt):
            pred_id = value % encoding_value # i. '나머지' 구하기. /21.4.11.10:01. 
            gt_id = int((value - pred_id)/encoding_value) # i. '몫' 구하기. pred_id 빼줄필요 없고, int()대신 //써도되지. gt_id = value//encoding_value 이렇게. /21.4.11.9:58. 
            if not gt_id in args.evalLabels:
                printError("Unknown label with id {:}".format(gt_id))
            # i. 이렇게해줬기때매 컨퓨젼매트릭스에서 row 가 gt 고 column 이 pred 가 되는거지. confMatrix[pred_id][gt_id]+=c 로 했으면 반대였겠지. /21.4.11.10:05. 
            confMatrix[gt_id][pred_id] += c 
        

    if args.evalInstLevelScore:
        # Generate category masks
        predCategoryMasksJ = {}
        for category in instanceStats["categories"]:
            predCategoryMasksJ[category] = np.in1d( predictionNp , instanceStats["categories"][category]["labelIds"] ).reshape(predictionNp.shape) 

        # i.21.3.28.23:44) TODO Q: instanceNp 는 단지 ~~_instanceIds.png 를 넘파이어레이로 만든것일뿐일텐데,
        #  인스턴스id 가 1000 일수도 있는데...??? >1000 이아니고 >999 로 해줘야하지않나??? 뭐 일단 cityscapes 나 내플젝에서는 상관없을것같지만. 
        instList = np.unique(instanceNp[instanceNp > 1000]) 
        for instId in instList:
            labelId = int(instId/1000)
            label = id2label[ labelId ]
            if label.ignoreInEval:
                continue

            # i. 좌변원래변수명은 mask. 
            #  명확의미위해 (변수명 gtInstMaskJ 의)앞에 gt 붙여줬지만, 사실 여기선 모델의 아웃풋중 sem seg 정보만 이용중이라
            #  인스턴스의 mask 라면 gt (~~_instanceIds.png) 에서부터 온 정보밖에없음. /21.3.30.9:44. 
            gtInstMaskJ = instanceNp==instId  
            instSize = np.count_nonzero( gtInstMaskJ )

            # i. instanceNp 는 ~~_instanceIds.png 의 넘파이어레이. 인스턴스id 들의 정보가 들어있음. 
            #  predictionNp 는 모델의 프레딕션결과중 "sem_seg" 정보를 이용한, (임시폴더의) ~~_pred.png 의 넘파이어레이. 
            #  predictionNp 에는 인스턴스id 가 아닌, 클래스id 들의 정보가 담겨있음. /21.3.29.0:52. -> 틀린내용 수정. /21.3.30.0:29. 
            tp = np.count_nonzero( predictionNp[gtInstMaskJ] == labelId ) 
            fn = instSize - tp

            # i. avgClassSize, tp, fn 관련해 원래파일(evalPixelLevelSemanticLabeling.py)의 이위치에남긴 코멘트 참고. /21.4.11.9:04. 
            weight = args.avgClassSize[label.name] / float(instSize)
            tpWeighted = float(tp) * weight
            fnWeighted = float(fn) * weight

            instanceStats["classes"][label.name]["tp"]         += tp
            instanceStats["classes"][label.name]["fn"]         += fn
            instanceStats["classes"][label.name]["tpWeighted"] += tpWeighted
            instanceStats["classes"][label.name]["fnWeighted"] += fnWeighted

            category = label.category
            if category in instanceStats["categories"]:
                catTp = 0
                catTp = np.count_nonzero( np.logical_and( gtInstMaskJ , predCategoryMasksJ[category] ) )
                catFn = instSize - catTp

                catTpWeighted = float(catTp) * weight
                catFnWeighted = float(catFn) * weight

                instanceStats["categories"][category]["tp"]         += catTp
                instanceStats["categories"][category]["fn"]         += catFn
                instanceStats["categories"][category]["tpWeighted"] += catTpWeighted
                instanceStats["categories"][category]["fnWeighted"] += catFnWeighted

    if args.evalPixelAccuracy:
        notIgnoredLabels = [l for l in args.evalLabels if not id2label[l].ignoreInEval]
        notIgnoredPixels = np.in1d( groundTruthNp , notIgnoredLabels , invert=True ).reshape(groundTruthNp.shape)
        erroneousPixels = np.logical_and( notIgnoredPixels , ( predictionNp != groundTruthNp ) )
        perImageStats[predictionImgFileName] = {}
        perImageStats[predictionImgFileName]["nbNotIgnoredPixels"] = np.count_nonzero(notIgnoredPixels)
        perImageStats[predictionImgFileName]["nbCorrectPixels"]    = np.count_nonzero(erroneousPixels)

    return nbPixels

# The main method
def main():
    global args
    argv = sys.argv[1:]

    predictionImgList = []
    groundTruthImgList = []

    # the image lists can either be provided as arguments
    if (len(argv) > 3):
        for arg in argv:
            if ("gt" in arg or "groundtruth" in arg):
                groundTruthImgList.append(arg)
            elif ("pred" in arg):
                predictionImgList.append(arg)
    # however the no-argument way is prefered
    elif len(argv) == 0:
        # use the ground truth search string specified above
        groundTruthImgList = glob.glob(args.groundTruthSearch)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append( getPrediction(args,gt) )

    # evaluate
    evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == "__main__":
    main()
