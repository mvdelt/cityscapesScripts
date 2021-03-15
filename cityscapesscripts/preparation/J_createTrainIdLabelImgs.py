#!/usr/bin/python


#######################################################################################################################
# i.21.3.10.20:15) 
#  ~~polygons.json 으로부터, (내플젝에 맞게) ~~labelTrainIds.png 만들어주기위한 파일.
#  ~~labelTrainIds.png 는 굳이 필요없는줄 알았는데
#  (~~instanceIds.png 와 어노json 이렇게 두개에 사실상 모든 정보 담겨있음),
#  이유는 모르겟지만 Det2 에서 "sem_seg_file_name" 를 필요로해서,
#  바로 그것에 해당하는 ~~labelTrainIds.png 를 내플젝에맞게 만들어주려함.
#  (Det2 문서 Use Builtin Datasets 에서 cityscapes 데이터셋구조 준비할때
#   createTrainIdLabelImgs.py 써서 ~~labelTrainIds.png 를 만들어주길래
#   굳이 이거 필요없는데 왜만들어주지 문서가 잘못됏네 라고 생각햇는데, 뭐.. 필요한가봄.)
#
# i.21.3.10.23:38) ->뭐지?? load_cityscapes_panoptic 함수
#  에 대응되는 COCO 함수인 load_coco_panoptic_json 에서는 "sem_seg_file_name" 정보 안넣어주는데???
#
#######################################################################################################################


# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode ground truth classes(근데 그냥 id 가 아니고 트레이닝용 id 인거지. labels.py 에 설정해놓은.).
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labels.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the converter tool in 'json2labelImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2labelImg import json2labelImg

# The main method
def main():
    # # Where to look for Cityscapes
    # if 'CITYSCAPES_DATASET' in os.environ:
    #     cityscapesPath = os.environ['CITYSCAPES_DATASET']
    # else:
    #     cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    # # how to search for all ground truth
    # searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" ) # i. ex: ~~\gtFine\val\frankfurt\~~polygons.json
    # searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # # search files
    # filesFine = glob.glob( searchFine )
    # filesFine.sort()
    # filesCoarse = glob.glob( searchCoarse )
    # filesCoarse.sort()

    # # concatenate fine and coarse
    # files = filesFine + filesCoarse
    # # files = filesFine # use this line if fine is enough for now.



    # Where to look for Cityscapes
    # i.21.3.11.12:45) 기존의 convertTestJ 폴더에서 panopticSeg_dentPanoJ 로 폴더명 바꿨고, 그안에 gt 및 inputOriPano 이렇게 두개 폴더 다시 만들어줬음.
    #  따라서 ~~polygons.json 경로 바꼈음. 바뀐 ~~polygons.json 경로 ex: panopticSeg_dentPanoJ\gt\train\imp2_1_polygons.json
    # MYROOTDIRPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ" # ~~polygons.json 경로 ex: convertTestJ\train\imp2_1_polygons.json  # <-요건 기존경로.
    # i.21.3.14.22:41) 코랩컴에서의 경로로 수정. 내 구글드라이브에 커스텀데이터 올려놓고, 코랩컴에서 구글드라이브의 압축파일을 (코랩컴의 디렉토리에다가)압축풀어서 사용할거니까.
    #  즉, 구글코랩에서 구글드라이브 연동해서 돌리는걸 가정한것임. 뭐 사실상 코랩에서만 할테니까 일단은.
    MYROOTDIRPATH_J = "/content/datasetsJ/panopticSeg_dentPanoJ"
    

    # how to search for all ground truth(i. ~~polygons.json)
    forSearchAllPolygonsJson = os.path.join(MYROOTDIRPATH_J, "gt", "*", "*_polygons.json")
    # search files
    polygonsjson_path_list = glob.glob(forSearchAllPolygonsJson)

    files = polygonsjson_path_list


    # quit if we did not find anything
    if not files:
        printError( "j) Did not find any files(~~polygons.json)!!!" )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_labelTrainIds.png" )

        # do the conversion
        try:
            json2labelImg( f , dst , "trainIds" )
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# call the main
if __name__ == "__main__":
    main()
