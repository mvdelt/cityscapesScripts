#!/usr/bin/env python3

# i.21.3.7.22:51) createPanopticImgs.py 를 내플젝에 맞게 수정한 파일.

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



CS_ROOTDIRPATH_J = r"C:\Users\starriet\Downloads\convertTestJ"


# The main method
def convert2panoptic(cityscapesPath=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if cityscapesPath is None:
        cityscapesPath = CS_ROOTDIRPATH_J

    if outputFolder is None:
        outputFolder = cityscapesPath

    categories = [] # i. ######################################
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    for setName in setNames:
        # how to search for all ground truth
        searchFine   = os.path.join(cityscapesPath, setName, "*_instanceIds.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "j) Did not find any files for {} set using matching pattern {} !!!".format(setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

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

            originalFormat = np.array(Image.open(f))  # i. f 는 path/to/~~instanceIds.png

            fileName = os.path.basename(f)

            # imageId = fileName.replace("_gtFine_instanceIds.png", "") # i. 변경필요 ###########
            # inputFileName = fileName.replace("_instanceIds.png", "_leftImg8bit.png") # i. 변경필요 ###########
            # outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")



            # fileName ex: imp2_0_instanceIds.png, imp4_120_instanceIds.png   (imp{A}_{00B}_instanceIds.png 이런식. A:2,3,4, B:0~3자리수)
            # i.21.3.8.오후쯤) ->여기서 A00B 이런식으로 이미지id 만들어줄거임. B가 두자리면 A0bb, 세자리면 Abbb 이런식으로.
            implDatasetGroupNumJ = fileName[3] # "2", "4"
            implSubNumJ = fileName[len("impX_"):-len("_instanceIds.png")] # "0", "120"

            imageIdJ = implDatasetGroupNumJ + (3-len(implSubNumJ))*"0" + implSubNumJ # "2000", "4120"   (A00B 이런식)
            inputImgFileNameJ = fileName.replace("_instanceIds.png", ".jpg")
            outAnnoPngNameJ = fileName.replace("_instanceIds.png", "_panopticAnno.png")



            # image entry, id for image is its filename without extension
            images.append({"id": imageIdJ,
                           "width": int(originalFormat.shape[1]),
                           "height": int(originalFormat.shape[0]),
                           "file_name": inputImgFileNameJ})

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            segmentIds = np.unique(originalFormat)
            segmInfo = []
            # i.21.3.8.00:25) 여기서 segmentIds 를 z-order에 맞게 정렬해줘야겠네.
            #  ->아니지. 지금 originalFormat(~~instanceIds.png 를 읽어들인거)은 이미 내가정해준 zorder대로 그려진상태니까
            #    이제는 zorder 상관할필요가 없겠네. 그냥 2차원 이미지일 뿐이니까.
            for segmentId in segmentIds:
                if segmentId < 1000:
                    semanticId = segmentId
                    isCrowd = 1
                else:
                    semanticId = segmentId // 1000
                    isCrowd = 0
                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval:
                    continue
                if not labelInfo.hasInstances:
                    isCrowd = 0

                mask = originalFormat == segmentId # i. TODO 넘파이 문법 복습필요.

                # color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256] 
                # i.21.3.8.22:28)->요게 기존 코드. 잘못됐음. cityscapes 데이터셋은 클래스가 35갠가 뿐이라 이렇게해도 문제되진 않지만, 
                #  만약 클래스갯수가 65개고 인스턴스갯수가 엄청많다거나, 클래스갯수가 66개 이상이된다거나 하면 문제됨. 현 cityscapes 의 id정해주는방식이라면.
                #  (참고1: 256^2=65536) 
                #  (참고2: COCO panoptic 형식에서 id=R+G*256+B*256^2, RGB는 어노png파일의 각 픽셀의 값.)
                color = [segmentId%256, segmentId%(256*256)//256, segmentId%(256*256*256)//(256*256)] 
                # i.21.3.8.22:28)->요게 내가 수정한거. 세번쨋놈은 그냥 segmentId//(256*256) 
                # 또는 segmentId//256/256 으로 해도 되지만(id가 256^3보다 작을것이라서), 일반화를 위해 저렇게 적었음. 규칙성도 눈에잘보이고.

                pan_format[mask] = color 
                # i. TODO ->pan_format 은 HxWx3일거고 mask 는 아마 HxW일텐데(확인필요). 넘파이 문법 복습필요. 
                #  그리고 color 는 list 일텐데, 일케해도되나?

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

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outAnnoPngNameJ))

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
                        default=["val", "train", "test"],
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.cityscapesPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
