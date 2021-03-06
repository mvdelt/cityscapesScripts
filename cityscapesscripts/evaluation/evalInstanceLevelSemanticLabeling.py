#!/usr/bin/python
#
# The evaluation script for instance-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# To run this script, make sure that your results contain text files
# (one for each test set image) with the content:
#   relPathPrediction1 labelIDPrediction1 confidencePrediction1
#   relPathPrediction2 labelIDPrediction2 confidencePrediction2
#   relPathPrediction3 labelIDPrediction3 confidencePrediction3
#   ...
#
# - The given paths "relPathPrediction" point to images that contain
# binary masks for the described predictions, where any non-zero is
# part of the predicted instance. The paths must not contain spaces,
# must be relative to the root directory and must point to locations
# within the root directory.
# - The label IDs "labelIDPrediction" specify the class of that mask,
# encoded as defined in labels.py. Note that the regular ID is used,
# not the train ID.
# - The field "confidencePrediction" is a float value that assigns a
# confidence score to the mask.
#
# Note that this tool creates a file named "gtInstances.json" during its
# first run. This file helps to speed up computation and should be deleted
# whenever anything changes in the ground truth annotations or anything
# goes wrong.

# python imports
from __future__ import print_function, absolute_import, division
import os, sys
import fnmatch
from copy import deepcopy

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import *
from cityscapesscripts.evaluation.instances2dict import instances2dict


###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
# Please read the instructions above for a description of
# the result file.
#
# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable CITYSCAPES_RESULTS
#   - environment variable CITYSCAPES_DATASET/results
#   - ../../results/"
# (Remember to set the variables using "export CITYSCAPES_<VARIABLE>=<path>".)
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
# <city>_123456_123456*.txt
# for a ground truth filename
# <city>_123456_123456_gtFine_instanceIds.png
def getPrediction( groundTruthFile , args ):
    # determine the prediction path, if the method is first called
    if not args.predictionPath:
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        elif 'CITYSCAPES_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['CITYSCAPES_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        args.predictionPath = os.path.abspath(rootPath)

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    csFile = getCsFileInfo(groundTruthFile)
    filePattern = "{}_{}_{}*.txt".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile


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

# Parameters that should be modified by user
args.exportFile         = os.path.join( args.cityscapesPath , "evaluationResults" , "resultInstanceLevelSemanticLabeling.json" )
args.groundTruthSearch  = os.path.join( args.cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_instanceIds.png" )
# i.21.3.30.15:45) ->args.groundTruthSearch 는 이 파일의 main() 메서드에서만 사용함.
#  지금 Det2 의 cityscapes_evaluation.py 에서는 이 파일의 evaluateImgLists 를 사용하고있기때문에 
#  args.groundTruthSearch 값을 내플젝에맞게 안바꿔줘도 잘 작동하는것임.
#  (Det2 에서 이 파일의 evaluateImgLists 사용해주기 전에 gt 파일들을 이미 찾아서 넣어주고있지. 난 그부분을 내플젝에맞게 바꿔준거고.) 


# overlaps for evaluation
args.overlaps           = np.arange(0.5,1.,0.05)
# minimum region size for evaluation [pixels]
args.minRegionSizes     = np.array( [ 100 , 1000 , 1000 ] )
# distance thresholds [m]
args.distanceThs        = np.array( [  float('inf') , 100 , 50 ] )
# distance confidences
args.distanceConfs      = np.array( [ -float('inf') , 0.5 , 0.5 ] )

args.gtInstancesFile    = os.path.join(os.path.dirname(os.path.realpath(__file__)),'gtInstances.json')
args.distanceAvailable  = False
args.JSONOutput         = True
args.quiet              = False
args.csv                = False
args.colorized          = True
args.instLabels         = []

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


# Determine the labels that have instances
def setInstanceLabels(args):
    args.instLabels = []
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            args.instLabels.append(label.name)

# Read prediction info
# imgFile, predId, confidence
def readPredInfo(predInfoFileName,args):
    predInfo = {}
    if (not os.path.isfile(predInfoFileName)):
        printError("Infofile '{}' for the predictions not found.".format(predInfoFileName))
    with open(predInfoFileName, 'r') as f:
        for line in f: # i. "{} {} {}\n".format(os.path.basename(png_filename), class_id, score) <-Det2의 cityscapes_evaluation.py 에서는. 
            splittedLine         = line.split(" ")
            if len(splittedLine) != 3:
                printError( "Invalid prediction file. Expected content: relPathPrediction1 labelIDPrediction1 confidencePrediction1" )
            if os.path.isabs(splittedLine[0]):
                printError( "Invalid prediction file. First entry in each line must be a relative path." )

            # i. 각각의 인스턴스의 mask 가 그려져있는 png파일의 경로. /21.3.31.11:22. 
            filename             = os.path.join( os.path.dirname(predInfoFileName),splittedLine[0] )
            filename             = os.path.abspath( filename )

            # check if that file is actually somewhere within the prediction root
            if os.path.commonprefix( [filename,args.predictionPath] ) != args.predictionPath:
                printError( "Predicted mask {} in prediction text file {} points outside of prediction path.".format(filename,predInfoFileName) )

            imageInfo            = {}
            imageInfo["labelID"] = int(float(splittedLine[1]))
            imageInfo["conf"]    = float(splittedLine[2]) # i. 인스턴스 디텍션 스코어. /21.3.31.11:16. 
            predInfo[filename]   = imageInfo

    return predInfo

# Routine to read ground truth image
def readGTImage(gtImageFileName,args):
    return Image.open(gtImageFileName)

# either read or compute a dictionary of all ground truth instances
def getGtInstances(groundTruthList,args):
    gtInstances = {}
    # if there is a global statistics json, then load it
    if (os.path.isfile(args.gtInstancesFile)):
        if not args.quiet:
            print("Loading ground truth instances from JSON.")
        with open(args.gtInstancesFile) as json_file:
            gtInstances = json.load(json_file)
    # otherwise create it
    else:
        if (not args.quiet):
            print("Creating ground truth instances from png files.")
        gtInstances = instances2dict(groundTruthList,not args.quiet)
        writeDict2JSON(gtInstances, args.gtInstancesFile)

    return gtInstances

# Filter instances, ignore labels without instances
def filterGtInstances(singleImageInstances,args):
    instanceDict = {}
    for labelName in singleImageInstances:
        if not labelName in args.instLabels:
            continue
        instanceDict[labelName] = singleImageInstances[labelName]
    return instanceDict

# match ground truth instances with predicted instances
def matchGtWithPreds(predictionList,groundTruthList,gtInstances,args):
    matches = {}
    if not args.quiet:
        print("Matching {} pairs of images...".format(len(predictionList)))

    count = 0
    for (pred,gt) in zip(predictionList,groundTruthList):
        # key for dicts
        dictKey = os.path.abspath(gt)

        # Read input files
        gtImage  = readGTImage(gt,args)
        predInfo = readPredInfo(pred,args)

        # Get and filter ground truth instances
        unfilteredInstances = gtInstances[ dictKey ]
        curGtInstancesOrig  = filterGtInstances(unfilteredInstances,args)

        # Try to assign all predictions
        (curGtInstances,curPredInstances) = assignGt2Preds(curGtInstancesOrig, gtImage, predInfo, args)

        # append to global dict
        matches[ dictKey ] = {}
        matches[ dictKey ]["groundTruth"] = curGtInstances
        matches[ dictKey ]["prediction"]  = curPredInstances

        count += 1
        if not args.quiet:
            print("\rImages Processed: {}".format(count), end=' ')
            sys.stdout.flush()

    if not args.quiet:
        print("")

    return matches

# For a given frame, assign all predicted instances to ground truth instances
def assignGt2Preds(gtInstancesOrig, gtImage, predInfo, args):
    # In this method, we create two lists     # i. <- lists 가 아니고 dicts. /21.3.31.13:45. 
    #  - predInstances: contains all predictions and their associated gt
    #  - gtInstances:   contains all gt instances and their associated predictions
    predInstances    = {}
    predInstCount    = 0

    # Create a prediction array for each class
    for label in args.instLabels:
        predInstances[label] = []

    # We already know about the gt instances
    # Add the matching information array
    gtInstances = deepcopy(gtInstancesOrig)
    for label in gtInstances:
        for gt in gtInstances[label]:
            gt["matchedPred"] = []

    # Make the gt a numpy array
    gtNp = np.array(gtImage)

    # Get a mask of void labels in the groundtruth
    voidLabelIDList = []
    for label in labels:
        if label.ignoreInEval:
            voidLabelIDList.append(label.id)    
    # i.21.3.31.11:52) TODO 이거 잘못된듯!!!!!
    #  voidLabelIDList 엔 ignoreInEval 이 True 인 label 의 label.id 들이 있는데 gtNp 엔 인스턴스id 들이 담겨있잖아. (gtNp 는 ~~_instanceIds.png 에서 온거임)
    #  stuff 들은 label.id 와 인스턴스id 가 같지만, thing 들은 다르지. 
    #  따라서, 만약 thing 중에서 ignoreInEval 이 True 인게 있다면, 이 코드로는 gtNp 에서 해당 thing 들을 잡아내질 못함!!
    #  실제로 labels.py 의 labels(cityscapes 에서 제시하는 기본값) 보면, thing 중에서도 ignoreInEval 이 True 인것들이 있음. (id 29 'caravan', id 30 'trailer'.) 
    #
    #  TODO 따라서, 내생각엔 아래 두줄로 수정해야함. arr=arr//1000 이 문법 되나 확인필요. 
    #  gtNp_labelIdsJ = np.copy(gtNp) // 1000
    #  boolVoid = np.in1d(gtNp_labelIdsJ, voidLabelIDList).reshape(gtNp.shape) 
    boolVoid = np.in1d(gtNp, voidLabelIDList).reshape(gtNp.shape) 

    # Loop through all prediction masks
    for predImageFile in predInfo:
        # Additional prediction info
        labelID  = predInfo[predImageFile]["labelID"]
        predConf = predInfo[predImageFile]["conf"]

        # label name
        labelName = id2label[int(labelID)].name

        # maybe we are not interested in that label
        # i. Det2 에선 이미 프레딕션결과중 인스턴스만 predImageFile 이 된거라서,
        #  요 if 문에 해당되어 continue 되는경우가 없을거임. 
        if not labelName in args.instLabels: 
            raise ValueError(f'j) 뭔가이상함!!!! {labelName} 이 stuff 인데, Det2 에선 이미 프레딕션결과중 인스턴스만 predImageFile 이 된건데??') # i. /21.3.31.13:15. 
            continue

        # Read the mask
        predImage = Image.open(predImageFile)
        predImage = predImage.convert("L")
        predNp    = np.array(predImage)

        # make the image really binary, i.e. everything non-zero is part of the prediction
        boolPredInst   = predNp != 0
        predPixelCount = np.count_nonzero( boolPredInst )

        # skip if actually empty
        if not predPixelCount:
            continue

        # The information we want to collect for this instance
        predInstance = {}
        predInstance["imgName"]          = predImageFile
        predInstance["predID"]           = predInstCount
        predInstance["labelID"]          = int(labelID)
        predInstance["pixelCount"]       = predPixelCount
        predInstance["confidence"]       = predConf
        # Determine the number of pixels overlapping void
        predInstance["voidIntersection"] = np.count_nonzero( np.logical_and(boolVoid, boolPredInst) )

        # A list of all overlapping ground truth instances
        matchedGt = []

        # Loop through all ground truth instances with matching label
        # This list contains all ground truth instances that distinguish groups
        # We do not know, if a certain instance is actually a single object or a group
        # e.g. car or cargroup
        # However, for now we treat both the same and do the rest later
        for (gtNum,gtInstance) in enumerate(gtInstancesOrig[labelName]):

            intersection = np.count_nonzero( np.logical_and( gtNp == gtInstance["instID"] , boolPredInst) )

            # If they intersect add them as matches to both dicts
            if (intersection > 0):
                gtCopy   = gtInstance.copy()
                predCopy = predInstance.copy()

                # let the two know their intersection
                gtCopy["intersection"]   = intersection
                predCopy["intersection"] = intersection

                # append ground truth to matches
                matchedGt.append(gtCopy)
                # append prediction to ground truth instance
                gtInstances[labelName][gtNum]["matchedPred"].append(predCopy)

        predInstance["matchedGt"] = matchedGt
        predInstCount += 1
        predInstances[labelName].append(predInstance)

    return (gtInstances,predInstances)


def evaluateMatches(matches, args):
    # In the end, we need two vectors for each class and for each overlap
    # The first vector (y_true) is binary and is 1, where the ground truth says true,
    # and is 0 otherwise.
    # The second vector (y_score) is float [0...1] and represents the confidence of
    # the prediction.
    #
    # We represent the following cases as:
    #                                       | y_true |   y_score
    #   gt instance with matched prediction |    1   | confidence
    #   gt instance w/o  matched prediction |    1   |     0.0
    #          false positive prediction    |    0   | confidence
    #
    # The current implementation makes only sense for an overlap threshold >= 0.5,
    # since only then, a single prediction can either be ignored or matched, but
    # never both. Further, it can never match to two gt instances.
    # For matching, we vary the overlap and do the following steps:
    #   1.) remove all predictions that satisfy the overlap criterion with an ignore region (either void or *group)
    #   2.) remove matches that do not satisfy the overlap
    #   3.) mark non-matched predictions as false positive

    # AP
    overlaps  = args.overlaps
    # region size
    minRegionSizes = args.minRegionSizes
    # distance thresholds
    distThs   = args.distanceThs
    # distance confidences
    distConfs = args.distanceConfs
    # only keep the first, if distances are not available
    if not args.distanceAvailable:
        minRegionSizes = [ minRegionSizes[0] ]
        distThs        = [ distThs       [0] ]
        distConfs      = [ distConfs     [0] ]

    # last three must be of same size
    if len(distThs) != len(minRegionSizes):
        printError("Number of distance thresholds and region sizes different")
    if len(distThs) != len(distConfs):
        printError("Number of distance thresholds and confidences different")

    # Here we hold the results
    # First dimension is class, second overlap
    ap = np.zeros( (len(distThs) , len(args.instLabels) , len(overlaps)) , np.float )

    for dI,(minRegionSize,distanceTh,distanceConf) in enumerate(zip(minRegionSizes,distThs,distConfs)):
        for (oI,overlapTh) in enumerate(overlaps):
            for (lI,labelName) in enumerate(args.instLabels):
                y_true   = np.empty( 0 )
                y_score  = np.empty( 0 )
                # count hard false negatives
                hardFns  = 0
                # found at least one gt and predicted instance?
                haveGt   = False
                havePred = False

                for img in matches:
                    predInstances = matches[img]["prediction" ][labelName]
                    gtInstances   = matches[img]["groundTruth"][labelName]
                    # filter groups in ground truth
                    gtInstances   = [ gt for gt in gtInstances if gt["instID"]>=1000 and gt["pixelCount"]>=minRegionSize and gt["medDist"]<=distanceTh and gt["distConf"]>=distanceConf ]

                    if gtInstances:
                        haveGt = True
                    if predInstances:
                        havePred = True

                    curTrue  = np.ones ( len(gtInstances) )
                    curScore = np.ones ( len(gtInstances) ) * (-float("inf"))
                    curMatch = np.zeros( len(gtInstances) , dtype=np.bool )

                    # collect matches
                    for (gtI,gt) in enumerate(gtInstances):
                        foundMatch = False
                        for pred in gt["matchedPred"]:
                            overlap = float(pred["intersection"]) / (gt["pixelCount"]+pred["pixelCount"]-pred["intersection"])
                            if overlap > overlapTh:
                                # the score
                                confidence = pred["confidence"]

                                # if we already hat a prediction for this groundtruth
                                # the prediction with the lower score is automatically a false positive
                                if curMatch[gtI]:
                                    maxScore = max( curScore[gtI] , confidence )
                                    minScore = min( curScore[gtI] , confidence )
                                    curScore[gtI] = maxScore
                                    # append false positive
                                    curTrue  = np.append(curTrue,0)
                                    curScore = np.append(curScore,minScore)
                                    curMatch = np.append(curMatch,True)
                                # otherwise set score
                                else:
                                    foundMatch = True
                                    curMatch[gtI] = True
                                    curScore[gtI] = confidence

                        if not foundMatch:
                            hardFns += 1

                    # remove non-matched ground truth instances
                    curTrue  = curTrue [ curMatch==True ]
                    curScore = curScore[ curMatch==True ]

                    # i.21.3.31.15:24) TODO 이부분 좀 잘못된듯!!!! 
                    #  nbIgnorePixels 를 overlap 계산시에 사용해주고,
                    #  if not foundGt 일때는 nbIgnorePixels 이용하지 말고 그냥 바로 false positive 로 해줘야하지않나??? 
                    #  overlap 계산시엔 분모계산시 nbIgnorePixels 값도 빼주도록 이렇게 수정하고말야: 
                    #  overlap = float(gt["intersection"]) / (gt["pixelCount"]+pred["pixelCount"]-gt["intersection"]  -nbIgnorePixels) 
                    #  
                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        foundGt = False
                        for gt in pred["matchedGt"]:
                            overlap = float(gt["intersection"]) / (gt["pixelCount"]+pred["pixelCount"]-gt["intersection"])
                            if overlap > overlapTh:
                                foundGt = True
                                break
                        if not foundGt:
                            # collect number of void and *group pixels
                            nbIgnorePixels = pred["voidIntersection"]
                            for gt in pred["matchedGt"]:
                                # group?
                                if gt["instID"] < 1000:  # i. TODO 지금 이 gt 는 stuff 랑 ignoreInEval=True 는 제외된건가?? 그게아니라면, group 중에서 ignoreInEval=True 인거는 이미 pred["voidIntersection"] 에 포함돼있잖아!!
                                    nbIgnorePixels += gt["intersection"]
                                # small ground truth instances
                                if gt["pixelCount"] < minRegionSize or gt["medDist"]>distanceTh or gt["distConf"]<distanceConf:
                                    nbIgnorePixels += gt["intersection"]
                            proportionIgnore = float(nbIgnorePixels)/pred["pixelCount"]
                            # if not ignored
                            # append false positive
                            if proportionIgnore <= overlapTh:
                                curTrue = np.append(curTrue,0)
                                confidence = pred["confidence"]
                                curScore = np.append(curScore,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,curTrue)
                    y_score = np.append(y_score,curScore)

                # compute the average precision
                if haveGt and havePred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    scoreArgSort      = np.argsort(y_score)
                    yScoreSorted      = y_score[scoreArgSort]
                    yTrueSorted       = y_true[scoreArgSort]
                    yTrueSortedCumsum = np.cumsum(yTrueSorted)

                    # unique thresholds
                    (thresholds,uniqueIndices) = np.unique( yScoreSorted , return_index=True )

                    # since we need to add an artificial point to the precision-recall curve
                    # increase its length by 1
                    nbPrecRecall = len(uniqueIndices) + 1

                    # prepare precision recall
                    nbExamples     = len(yScoreSorted)
                    nbTrueExamples = yTrueSortedCumsum[-1]
                    precision      = np.zeros(nbPrecRecall)
                    recall         = np.zeros(nbPrecRecall)

                    # deal with the first point
                    # only thing we need to do, is to append a zero to the cumsum at the end.
                    # an index of -1 uses that zero then
                    yTrueSortedCumsum = np.append( yTrueSortedCumsum , 0 )

                    # deal with remaining
                    for idxRes,idxScores in enumerate(uniqueIndices):
                        cumSum = yTrueSortedCumsum[idxScores-1]
                        tp = nbTrueExamples - cumSum
                        fp = nbExamples     - idxScores - tp
                        fn = cumSum + hardFns
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idxRes] = p
                        recall   [idxRes] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    # integration is performed via zero order, or equivalently step-wise integration
                    # first compute the widths of each step:
                    # use a convolution with appropriate kernel, manually deal with the boundaries first
                    recallForConv = np.copy(recall)
                    recallForConv = np.append( recallForConv[0] , recallForConv )
                    recallForConv = np.append( recallForConv    , 0.            )

                    stepWidths = np.convolve(recallForConv,[-0.5,0,0.5],'valid')

                    # integrate is now simply a dot product
                    apCurrent = np.dot( precision , stepWidths )

                elif haveGt:
                    apCurrent = 0.0
                else:
                    apCurrent = float('nan')
                ap[dI,lI,oI] = apCurrent

    return ap

def computeAverages(aps,args):
    # max distance index
    dInf  = np.argmax( args.distanceThs )
    d50m  = np.where( np.isclose( args.distanceThs ,  50. ) )
    d100m = np.where( np.isclose( args.distanceThs , 100. ) )
    o50   = np.where(np.isclose(args.overlaps,0.5  ))

    avgDict = {}
    avgDict["allAp"]       = np.nanmean(aps[ dInf,:,:  ])
    avgDict["allAp50%"]    = np.nanmean(aps[ dInf,:,o50])

    if args.distanceAvailable:
        avgDict["allAp50m"]    = np.nanmean(aps[ d50m,:,  :])
        avgDict["allAp100m"]   = np.nanmean(aps[d100m,:,  :])
        avgDict["allAp50%50m"] = np.nanmean(aps[ d50m,:,o50])

    avgDict["classes"]  = {}
    for (lI,labelName) in enumerate(args.instLabels):
        avgDict["classes"][labelName]             = {}
        avgDict["classes"][labelName]["ap"]       = np.average(aps[ dInf,lI,  :])
        avgDict["classes"][labelName]["ap50%"]    = np.average(aps[ dInf,lI,o50])
        if args.distanceAvailable:
            avgDict["classes"][labelName]["ap50m"]    = np.average(aps[ d50m,lI,  :])
            avgDict["classes"][labelName]["ap100m"]   = np.average(aps[d100m,lI,  :])
            avgDict["classes"][labelName]["ap50%50m"] = np.average(aps[ d50m,lI,o50])

    return avgDict

def printResults(avgDict, args):
    sep     = (","         if args.csv       else "")
    col1    = (":"         if not args.csv   else "")
    noCol   = (colors.ENDC if args.colorized else "")
    bold    = (colors.BOLD if args.colorized else "")
    lineLen = 50
    if args.distanceAvailable:
        lineLen += 40

    print("")
    if not args.csv:
        print("#"*lineLen)
    line  = bold
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    if args.distanceAvailable:
        line += "{:>15}".format("AP_50m"    ) + sep
        line += "{:>15}".format("AP_100m"   ) + sep
        line += "{:>15}".format("AP_50%50m" ) + sep
    line += noCol
    print(line)
    if not args.csv:
        print("#"*lineLen)

    for (lI,labelName) in enumerate(args.instLabels):
        apAvg  = avgDict["classes"][labelName]["ap"]
        ap50o  = avgDict["classes"][labelName]["ap50%"]
        if args.distanceAvailable:
            ap50m  = avgDict["classes"][labelName]["ap50m"]
            ap100m = avgDict["classes"][labelName]["ap100m"]
            ap5050 = avgDict["classes"][labelName]["ap50%50m"]

        line  = "{:<15}".format(labelName) + sep + col1
        line += getColorEntry(apAvg , args) + sep + "{:>15.3f}".format(apAvg ) + sep
        line += getColorEntry(ap50o , args) + sep + "{:>15.3f}".format(ap50o ) + sep
        if args.distanceAvailable:
            line += getColorEntry(ap50m , args) + sep + "{:>15.3f}".format(ap50m ) + sep
            line += getColorEntry(ap100m, args) + sep + "{:>15.3f}".format(ap100m) + sep
            line += getColorEntry(ap5050, args) + sep + "{:>15.3f}".format(ap5050) + sep
        line += noCol
        print(line)

    allApAvg  = avgDict["allAp"]
    allAp50o  = avgDict["allAp50%"]
    if args.distanceAvailable:
        allAp50m  = avgDict["allAp50m"]
        allAp100m = avgDict["allAp100m"]
        allAp5050 = avgDict["allAp50%50m"]

    if not args.csv:
            print("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += getColorEntry(allApAvg , args) + sep + "{:>15.3f}".format(allApAvg)  + sep
    line += getColorEntry(allAp50o , args) + sep + "{:>15.3f}".format(allAp50o)  + sep
    if args.distanceAvailable:
        line += getColorEntry(allAp50m , args) + sep + "{:>15.3f}".format(allAp50m)  + sep
        line += getColorEntry(allAp100m, args) + sep + "{:>15.3f}".format(allAp100m) + sep
        line += getColorEntry(allAp5050, args) + sep + "{:>15.3f}".format(allAp5050) + sep
    line += noCol
    print(line)
    print("")

def prepareJSONDataForResults(avgDict, aps, args):
    JSONData = {}
    JSONData["averages"] = avgDict
    JSONData["overlaps"] = args.overlaps.tolist()
    JSONData["minRegionSizes"]      = args.minRegionSizes.tolist()
    JSONData["distanceThresholds"]  = args.distanceThs.tolist()
    JSONData["minStereoDensities"]  = args.distanceConfs.tolist()
    JSONData["instLabels"] = args.instLabels
    JSONData["resultApMatrix"] = aps.tolist()

    return JSONData

# Work through image list
def evaluateImgLists(predictionList, groundTruthList, args):
    # determine labels of interest
    setInstanceLabels(args)
    # get dictionary of all ground truth instances
    gtInstances = getGtInstances(groundTruthList,args)
    # match predictions and ground truth
    matches = matchGtWithPreds(predictionList,groundTruthList,gtInstances,args)
    writeDict2JSON(matches,"matches.json")
    # evaluate matches
    apScores = evaluateMatches(matches, args)
    # averages
    avgDict = computeAverages(apScores,args)
    # result dict
    resDict = prepareJSONDataForResults(avgDict, apScores, args)
    if args.JSONOutput:
        # create output folder if necessary
        path = os.path.dirname(args.exportFile)
        ensurePath(path)
        # Write APs to JSON
        writeDict2JSON(resDict, args.exportFile)

    if not args.quiet:
         # Print results
        printResults(avgDict, args)

    return resDict

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
            predictionImgList.append( getPrediction(gt,args) )

    # print some info for user
    print("Note that this tool uses the file '{}' to cache the ground truth instances.".format(args.gtInstancesFile))
    print("If anything goes wrong, or if you change the ground truth, please delete the file.")

    # evaluate
    evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == "__main__":
    main()
