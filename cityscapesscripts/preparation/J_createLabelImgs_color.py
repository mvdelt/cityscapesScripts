#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode ground truth classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 

# i.21.3.7.16:40쯤) 지금 이 파일은 왜만든거냐면, ~~polygons.json 파일로부터 ~~color.png 파일 만들기위함임.
#  labels.py 에 내가 정해준 값들중에서, id값(및 같은id인데 thing일경우 instance구분값도추가될수있지)말고 'color' 로 png파일 그려주려고.
#  시각적으로 쉽게 확인해보기위해서.
# i. With this tool, you can generate option
#   a) *color.png     : the class is encoded by its color

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
    


# i.21.3.14.22:48) 지금 다른파일들의 경로는 내로컬컴에서의 경로로 돼있던걸 구글드라이브에서의 경로로 바꿔주고있는데(이제 코랩에서 돌려볼거라), 
#  지금이파일은 그냥 시각화해서 체크만해볼목적으로 만들었던거라서 코랩에서 돌릴일 없을듯해서 경로는 걍 그대로(내로컬컴에서의 경로) 둠.
MYROOTDIRPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ" 
# i. 21.3.14.22:57) ~~polygons.json 경로 ex: ~~\panopticSeg_dentPanoJ\gt\train\imp2_1_polygons.json
forSearchAllPolygonsJson = os.path.join(MYROOTDIRPATH_J, "gt", "*", "*_polygons.json")
polygonsjson_path_list = glob.glob(forSearchAllPolygonsJson)
print(f'j) ~~polygons.json path list: {polygonsjson_path_list}')
print(f'j) type polygonsjson_path_list: {type(polygonsjson_path_list)}')


# The main method
def main():


    files = polygonsjson_path_list


    # quit if we did not find anything
    if not files:
        printError( "j) Did not find any files~!~!~!" )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_color.png" )

        # do the conversion
        try:
            json2labelImg( f , dst , "color" )
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
