#!/usr/bin/env python3

########################################################################################################################
#
# i.21.3.7.22:51) createPanopticImgs.py 를 내플젝에 맞게 수정한 파일.
# i.21.3.11.11:25) 이 파일이 하는일 설명추가: 
#  각 train, val 등의 폴더에 대해서, (여기서 '등'이라고 적은이유는 test폴더는 아직안만들어줘서.)
#  ~~instanceIds.png 들로부터 COCO panoptic 형식(어노png들과 어노json) 을 내뱉음.
#    이때 ~~instanceIds.png 의 '파일명' 도 정보의 일부로서 이용함!
#  cityscapes 데이터셋 방식은, 파일명이 {이미지id}_gtFine_instanceIds.png 이런식이라서,
#  파일명으로부터 이 어노png가 어떤 원 인풋이미지에 대한 어노정보인지 알수있음. 
#    그리고 cityscapesscripts 의 labels.py 에 적어둔 정보들도 사용하고.
#
########################################################################################################################

# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
#
# By default with this tool uses IDs specified in labels.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For cityscapes image_id has form <city>_123456_123456 and corresponds to the prefix
# of cityscapes image files.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import id2label, labels



# CS_ROOTDIRPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ"
# i.21.3.14.22:51) 코랩에서돌려주기위해 코랩컴에서의 경로로 바꿔줬음. 
#  물론 내 로컬컴에서 돌려서 파일들 생성한뒤에 구글드라이브에 올려(서 코랩으로 복사or압축풀어)도 되긴 하지.
CS_ROOTDIRPATH_J = "/content/datasetsJ/panopticSeg_dentPanoJ" 


# The main method
def convert2panoptic(cityscapesPath=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    print(f'j) <inputs shoud be> cityscapesPath:None, outputFolder:None, useTrainId:False, setNames:["train"]')
    print(f'j)   <actual inputs> cityscapesPath:{cityscapesPath}, outputFolder:{outputFolder}, useTrainId:{useTrainId}, setNames:{setNames}')
    # Where to look for Cityscapes
    if cityscapesPath is None:
        cityscapesPath = CS_ROOTDIRPATH_J

    if outputFolder is None:
        outputFolder = os.path.join(CS_ROOTDIRPATH_J, "gt")

    categories = [] # i. ######################################
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    # i. train, val 등의 폴더에 대해서.
    for setName in setNames:
        # how to search for all ground truth
        forSearchInstanceIdsPngJ   = os.path.join(cityscapesPath, "gt", setName, "*_instanceIds.png")
        # search files
        instanceIdsPngPath_list = glob.glob(forSearchInstanceIdsPngJ)
        instanceIdsPngPath_list.sort()

        files = instanceIdsPngPath_list
        # quit if we did not find anything
        if not files:
            printError(
                "j) Did not find any files for {} set using matching pattern {} !!!".format(setName, forSearchInstanceIdsPngJ)
            )
        # a bit verbose
        print("j) Converting {} annotation files(~~instanceIds.png) for {} set.".format(len(files), setName))

        trainIfSuffix = "_trainId" if useTrainId else ""
        # outputBaseFile = "cityscapes_panoptic_{}{}".format(setName, trainIfSuffix)
        outputBaseNameJ = "J_cocoformat_panoptic_{}{}".format(setName, trainIfSuffix)
        outAnnoJsonPathJ = os.path.join(outputFolder, "{}.json".format(outputBaseNameJ))
        print("Json file with the annotations in panoptic format will be saved in {}".format(outAnnoJsonPathJ))
        panopticFolder = os.path.join(outputFolder, outputBaseNameJ)
        if not os.path.isdir(panopticFolder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
            os.mkdir(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        images = [] # i. ######################################
        annotations = [] # i. ######################################
        for progress, f in enumerate(files):

            cs_annoPng_arrJ = np.array(Image.open(f))  # i. f 는 path/to/~~instanceIds.png
            print(f'j) cs_annoPng_arrJ.shape: {cs_annoPng_arrJ.shape}') # i. ex: (976, 1976)

            fileName = os.path.basename(f)

            # imageId = fileName.replace("_gtFine_instanceIds.png", "") # i. 변경필요 ###########
            # inputFileName = fileName.replace("_instanceIds.png", "_leftImg8bit.png") # i. 변경필요 ###########
            # outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")




            # # fileName ex: imp2_0_instanceIds.png, imp4_120_instanceIds.png   (imp{A}_{00B}_instanceIds.png 이런식. A:2,3,4, B:0~3자리수)
            # # i.21.3.8.오후쯤) ->여기서 A00B 이런식으로 이미지id 만들어줄거임. B가 두자리면 A0bb, 세자리면 Abbb 이런식으로.
            # # i.21.3.10.19:37) 굳이이렇게할필요없음. 이미지id 는 그냥 문자열 숫자열 섞여있어도 상관없음.
            # #  그리고 Det2 의 데이터셋레지스터하는 코드 사용하려면.. 이미지id를 베이스네임?같은식으로 좀 맞춰줘야해서.. 암튼 이코드부분 수정필요.
            # #    ->바로아래에서수정함/21.3.16.11:42.
            # implDatasetGroupNumJ = fileName[3] # "2", "4"
            # implSubNumJ = fileName[len("impX_"):-len("_instanceIds.png")] # "0", "120"

            # imageIdJ = implDatasetGroupNumJ + (3-len(implSubNumJ))*"0" + implSubNumJ # "2000", "4120"   (A00B 이런식)
            # inputImgFileNameJ = fileName.replace("_instanceIds.png", ".jpg")
            # outAnnoPngNameJ = fileName.replace("_instanceIds.png", "_panopticAnno.png")


            # i.21.3.16.11:37) 내가 기존엔 바로위코드처럼 이미지id 를 A00B이런식으로 만들어줬었는데, 
            #  굳이그럴필요없고 그냥 'imp2_0' 이런식으로 스트링으로 해줘도돼서, 그렇게하기로했음.
            imageIdJ = fileName[:-len("_instanceIds.png")]  # 'imp2_0' 이런식.
            inputImgFileNameJ = fileName.replace("_instanceIds.png", ".jpg")
            outAnnoPngNameJ = fileName.replace("_instanceIds.png", "_panopticAnno.png")




            # image entry, id for image is its filename without extension
            images.append({"id": imageIdJ,
                           "width": int(cs_annoPng_arrJ.shape[1]),
                           "height": int(cs_annoPng_arrJ.shape[0]),
                           "file_name": inputImgFileNameJ})

            coco_annoPng_arrJ = np.zeros(
                (cs_annoPng_arrJ.shape[0], cs_annoPng_arrJ.shape[1], 3), dtype=np.uint8  # i.21.3.9.8:04) Unsigned integer 0 to 255
            )
            print(f'j) coco_annoPng_arrJ.shape: {coco_annoPng_arrJ.shape}') # i. ex: (976, 1976, 3)

            segmentIds = np.unique(cs_annoPng_arrJ)  # i. ex: [   0    1    2    3    4    5    6 7000 7001 7002 7003 7004 7005 7006 8000 8001 8002 8003 9000] 
            print(f'j) segmentIds = np.unique(cs_annoPng_arrJ): {np.unique(cs_annoPng_arrJ)}')
            segmInfo = []
            # i.21.3.8.00:25) 여기서 segmentIds 를 z-order에 맞게 정렬해줘야겠네.
            #  ->아니지. 지금 cs_annoPng_arrJ(~~instanceIds.png 를 읽어들인거)은 이미 내가정해준 zorder대로 그려진상태니까
            #    이제는 zorder 상관할필요가 없겠네. 그냥 2차원 이미지일 뿐이니까.
            for segmentId in segmentIds:
                if segmentId < 1000:  # i. id값이 1000미만이면, stuff(COCO형식에서 stuff는 iscrowd의미없고 기본적으로 0임)이거나, iscrowd=1인 thing임. 참고로 iscrowd 는 COCO형식에 나오는 값./21.3.9.9:38.
                    semanticId = segmentId
                    isCrowd = 1
                else: # i. id값이 1000이상이면, iscrowd=0 인 thing임./21.3.9.9:38.
                    semanticId = segmentId // 1000 
                    isCrowd = 0
                
                # i.21.3.17.23:06) 내가 'unlabeled_Label' 을 없애줬고 백그라운드의 segmentId 값은 내가정해준대로 255 일것이므로, 
                #  바로아래의 labelInfo = id2label[semanticId] 가 실행되면 KeyError: 255 가 발생함.
                #  근데 어차피 기존 cityscapes 의 labels.py 대로라고 해도, label 의 ignoreInEval 이 True 일 경우 continue 해서 무시해주고있음(아래 보면 나오지).
                #  즉, 백그라운드는 걍 무시하면 됨. 따라서 semanticId(=segmentId) 값이 255이면 continue 해줌.
                if semanticId == 255:
                    continue

                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval: # i. <- 요거 기존코드인데, ignoreInEval 이 True 면 걍 continue 해서 무시하는것을 볼수있음. /21.3.17.23:14.
                    continue
                if not labelInfo.hasInstances: # i. stuff면, iscrowd 의미없고 기본적으로 0임.(COCO형식 참고하삼)/21.3.9.9:45.
                    isCrowd = 0

                mask = cs_annoPng_arrJ == segmentId 
                # print(f'j) mask.shape: {mask.shape}') # i. ex: (976, 1976)

                # color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256] 
                # i.21.3.8.22:28)->요게 기존 코드. 잘못됐음. cityscapes 데이터셋은 클래스가 35갠가 뿐이라 이렇게해도 문제되진 않지만, 
                #  만약 클래스갯수가 65개고 인스턴스갯수가 엄청많다거나, 클래스갯수가 66개 이상이된다거나 하면 문제됨. 현 cityscapes 의 id정해주는방식이라면.
                #  즉, 예를들어 id값이 65900(클래스번호65, 해당클래스의 901번째 인스턴스)라든가, id값이 66000(클래스번호66, 해당클래스의 0번째인스턴스)라든가 이럴경우 문제됨.
                #  (참고1: 256^2=65536) 
                #  (참고2: COCO panoptic 형식에서 id=R+G*256+B*256^2, RGB는 어노png파일의 각 픽셀의 값.)
                color = [segmentId%256, segmentId%(256*256)//256, segmentId%(256*256*256)//(256*256)] 
                print(f'j) id:{segmentId} -> color(RGB):{color}')
                # i.21.3.8.22:28)->요게 내가 수정한거. 세번쨋놈은 그냥 segmentId//(256*256) 
                # 또는 segmentId//256/256 으로 해도 되지만(id가 256^3보다 작을것이라서), 일반화를 위해 저렇게 적었음. 규칙성도 눈에잘보이고.
                #
                # i.21.3.10.19:44) cocodataset깃헙보면 cocoapi 말고도 panopticapi 라고 있음. 
                #  거기서  panopticapi.utils.id2rgb 함수가 바로 위의 변환이랑 내내 같은걸 해주고있음. (Det2 문서의 "pan_seg_file_name" 설명에 나오는 함수)
                #  거깄는 id2rgb 함수는 2차원 id맵(numpy어레이)을 넣으면 RGB맵으로 변환해주고, 그냥 하나의 id값만 넣으면 [R,G,B] 리스트를 반환해줌.

                coco_annoPng_arrJ[mask] = color # i. ->coco_annoPng_arrJ 은 HxWx3, mask 는 HxW. 그래도 상관없지.
                # ->color 는 list 지만, 일케해줘도 상관x. 넘파이어레이로 됨.

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segmInfo.append({"id": int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': imageIdJ,
                                'file_name': outAnnoPngNameJ,
                                "segments_info": segmInfo})

            Image.fromarray(coco_annoPng_arrJ).save(os.path.join(panopticFolder, outAnnoPngNameJ))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()

        print("\nSaving the json file {}".format(outAnnoJsonPathJ))
        d = {'images': images,
             'annotations': annotations,
             'categories': categories}
        with open(outAnnoJsonPathJ, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="cityscapesPath",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default=None,
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default=None,
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        # default=["val", "train", "test"],
                        default=["train"], # i. 일단 train 폴더만 만들어줘놔봤음./21.3.9.10:11
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.cityscapesPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
