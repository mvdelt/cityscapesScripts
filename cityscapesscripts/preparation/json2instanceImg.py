#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to instance images,
# where each pixel has an ID that represents the ground truth class and the
# individual instance of that class.
#
# The pixel values encode both, class and the individual instance.
# The integer part of a division by 1000 of each ID provides the class ID,
# as described in labels.py. The remainder is the instance ID. If a certain
# annotation describes multiple instances, then the pixels have the regular
# ID of that class.
#
# Example:
# Let's say your labels.py assigns the ID 26 to the class 'car'.
# Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
# A group of cars, where our annotators could not identify the individual
# instances anymore, is assigned to the ID 26.
#
# Note that not all classes distinguish instances (see labels.py for a full list).
# The classes without instance annotations are always directly encoded with
# their regular ID, e.g. 11 for 'building'.
#
# Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt

# Image processing
from PIL import Image
from PIL import ImageDraw

# cityscapes imports
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import labels, name2label


# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' Reads labels as polygons in JSON format and converts them to instance images,')
    print(' where each pixel has an ID that represents the ground truth class and the')
    print(' individual instance of that class.')
    print('')
    print(' The pixel values encode both, class and the individual instance.')
    print(' The integer part of a division by 1000 of each ID provides the class ID,')
    print(' as described in labels.py. The remainder is the instance ID. If a certain')
    print(' annotation describes multiple instances, then the pixels have the regular')
    print(' ID of that class.')
    print('')
    print(' Example:')
    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
    print(' A group of cars, where our annotators could not identify the individual')
    print(' instances anymore, is assigned to the ID 26.')
    print('')
    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
    print(' The classes without instance annotations are always directly encoded with')
    print(' their regular ID, e.g. 11 for "building".')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image
def createInstanceImage(annotation, encoding):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    # the background
    if encoding == "ids":
        
        backgroundId = name2label['unlabeled_Label'].id  # i.21.3.7.13:36) 내가 labels.py 에서 'unlabeled_Label' 이라고 바꿔줬었기때매 여기도 이렇게 해줌.
        
        # i.21.3.17.22:03) labels.py 에서 unlabeled_Label 를 아예 없애줘버렷기때매, 걍 내가 255로 직접 지정해줫음.
        #  기존에는 name2label['unlabeled_Label'].id 값이 0 이었기때문에 어노png에 백그라운드는 id값 0으로 그려졌었는데,
        #  기존엔 'unlabeled_Label' 도 클래스중 하나로 처리됐었기때문에 그렇게 하면 되지만,
        #  이제 내가 labels.py 의 카테고리 목록에서 unlabeled_Label 을 없애버렸기때문에 이 값을 무시하도록 되어있는 값으로 해줘야할듯함.
        #  아마 255가 무시하는 값일거임. 기본적으로 그냥 카테고리id 가 아닌, 카테고리의 trainId 를 사용하는게 표준적인 방식인것같은데(난 모르고 걍 카테고리id 사용했지만),
        #  labels.py 보면 기본의 'unlabeled' 카테고리 및 다른 무시해주는 카테고리들의 trainId 값이 255로 되어있고,
        #  Metadata 값 정해줄때도 ignore_label=255 로 해주고있고.
        #
        # i.21.3.18.10:53) 백그라운드의 id값은 뭐가돼도 상관없기는 함.
        #  ('unlabeled_Label'의 id값이랑은 같아야함. createPanopticImgs.py 에서 ~~instanceIds.png 의 id값들로 그에 해당하는 Label들의 정보를 확인하기때문에.) 
        #  왜 상관없냐면, 지금 여기서 ~~instanceIds.png 를 만들어주는건데(물론 다른 cityscapes어노png 들도 만들수있지만), 
        #  백그라운드 Label 의 'ignoreInEval' 값을 True 로 해줬을것인데,
        #  createPanopticImgs.py 에서 ~~instanceIds.png 로부터 coco어노png 를 만들어줄때 ignoreInEval 이 True 인 것들은 그려주지 않고 스킵하기 때문에.
        #    즉, 지금 다시 unlabeled_Label 사용해주기로했고 unlabeled_Label 의 id값이 0이지만,
        #  백그라운드의 id값은 뭘로해도 상관없긴함. 어차피 coco어노png 에는 안그려지니까(ignoreInEval 이 True 인 이상).
        #  그리고 Det2 에서는 'unlabeled'(내플젝에선 'unlabeled_Label') 는 아예 사용을 안함. 의미있는 카테고리들(foreground카테고리들)의 정보만 사용함.
        #    다만, 의미있는 카테고리(클래스)들은 id값이 0이면 안됨.
        #  왜냐면, createPanopticImgs.py 에서 ~~instanceIds.png 로부터 coco어노png 를 만들어줄때 백그라운드 픽셀을 [0,0,0] 으로 해주고있는데,
        #  id값이 0이면 256진법으로 RGB로 변환됐을때 [0,0,0] 이기때문에 백그라운드랑 값이 똑같아져버림!!
        #  (내가 mandible 의 id 를 0으로 했다가 백그라운드랑 똑같이 취급돼버렷지.)
        #    따라서 요 한줄은 코멘트아웃하고 다시 backgroundId = name2label['unlabeled_Label'].id 로 해줌.
        # backgroundId = 255 

    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId) # i. I (32-bit signed integer pixels)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( instanceImg )

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0

    # loop over all objects  
    # i.21.3.6.22:38) annotation.objects 는 리스트기때문에 원소들의 순서가 구분됨. 
    #  따라서 그 순서를 잘 정해주면, 지금 여기서 그 순서대로 그림을 그려주기때문에,
    #  출력되는 png 파일에 원하는순서대로 그림을 그려줄수있음. 
    #  (예를들어 A물체를 먼저그리고 그 위에 B물체를 그려야할경우 그 순서를 맞춰줘야한다는거지)
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if ( not label in name2label ) and label.endswith('group'):  # i. cityscapes 데이터셋에선 클래스명이 'group'으로 끝나면 뭐 그룹으로 봐주도록 하나봄. (예: polegroup) 근데 labels.py 에는 그런 클래스명이 polegroup 밖에 없네;
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between individual instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup and id != 255:
            id = id * 1000 + nbInstances[label]
            nbInstances[label] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue

        try:
            drawer.polygon( polygon, fill=id )
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
            raise

    return instanceImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the instance image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
def json2instanceImg(inJson,outImg,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    instanceImg = createInstanceImage( annotation , encoding )
    instanceImg.save( outImg )

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2instanceImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if len(args) == 0:
        printError( "Missing input json file" )
    elif len(args) == 1:
        printError( "Missing output image filename" )
    elif len(args) > 2:
        printError( "Too many arguments" )

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2instanceImg( inJson , outImg , 'trainIds' )
    else:
        json2instanceImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
