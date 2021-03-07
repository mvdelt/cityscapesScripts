#!/usr/bin/env python3
# i.21.3.7.15:51) -> shebang 에 대해 조사완료. 위와같이 하는게 젤 나음. python 말고 python3 이라고 하는게 좋음.
#  맨처음 #! 이후에는 full path 가 와야함. /usr/bin/env 라는 풀패스를 이용해서 env 라는 프로그램을 실행시키는거고,
#  env 라는걸 실행시키면서 아규먼트로 python3 라는것을 넣어주는거임.
#  그러면 env 는 environment variables 중에 $PATH 에 나열된 경로들중에서 맨 처음으로 python3 이 있는 경로를 이용해서 python3을 실행시킴.
#  즉, /usr/bin/env  python3  path/to/this/script.py 라고 커맨드창에 쳐주는것과 마찬가지인셈.
#  (만약 쉬뱅을 #!/usr/bin/python 이런식으로 해놧다면,
#  /usr/bin/python  path/to/this/script.py 이런식으로 커맨드창에 입력해서 실행해주는셈임.)


##########################################################################################################################
# i.21.3.7.15:59) 기존의 createTrainIdInstanceImgs.py 를 좀 변경해서, 
#  ~~polygons.json 파일로부터
#  ~~instanceTrainIds.png 말고 ~~instanceIds.png 들 만들어주려고(내 커스텀데이터셋으로) 지금 이 파일 작성중.
##########################################################################################################################


# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 

# i. 지금 이 파일에서 뭐하려는거냐면,
#  내 커스텀데이터셋으로 ~~instanceIds.png 들 만들어주려고 createTrainIdInstanceImgs.py 의 코드 수정해보려는중.
#  (createTrainIdInstanceImgs.py 에서는 ~~instanceTrainIds.png 를 만듦.)
# i. With this tool, you can generate
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID

#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#



# # python imports
# from __future__ import print_function, absolute_import, division
import os, glob, sys

# # cityscapes imports
# from cityscapesscripts.helpers.csHelpers import printError
# from cityscapesscripts.preparation.json2instanceImg import json2instanceImg
from json2instanceImg import json2instanceImg



POLYGONSJSON_DIRPATH_J = r"C:\Users\starriet\Downloads\convertTestJ"
forSearchAllPolygonsJson = os.path.join(POLYGONSJSON_DIRPATH_J, "*_polygons.json")
polygonsjson_path_list = glob.glob(forSearchAllPolygonsJson)
print(f'j) ~~polygons.json path list: {polygonsjson_path_list}')
print(f'j) type polygonsjson_path_list: {type(polygonsjson_path_list)}')



# The main method
def main():

    files = polygonsjson_path_list

    # quit if we did not find anything
    if not files:
        raise ValueError( "j) Did not find any files!!!" )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # create the output filename
        dst = f.replace( "_polygons.json" , "_instanceIds.png" ) # i. "_instanceTrainIds.png" 말고 "_instanceIds.png" 로 해줬음.

        # do the conversion
        try:
            json2instanceImg( f , dst , "ids" ) # i. 세번째인풋인 encoding 을 "trainIds" 말고 "ids" 로 해줬음.
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
