

########################################################################################################################
#
# i. 21.3.5.19:56) 
#  요약: COCO object detection 형식 -> cityscapes 의 ~~polygons.json 의 형식 으로 형식바꿔주는 코드. 
#  어떤 특정 어노테이션툴에서, 세그멘테이션 어노테이션한걸
#  COCO object detection 형식으로 내뱉었다고 했을때, 
#  그걸 cityscapes 의 ~~polygons.json 의 형식으로 바꿔주기 위한 코드임.
#  ~~polygons.json 만 있으면, cityscapesscripts 의 코드들을 사용해서 
#  원하는형식의 어노테이션png파일들 생성가능하고(cityscapes 에서 어노테이션png 파일들 종류가 몇개 있지),
#  createPanopticImgs.py 이용해서 ~~instanceIds.png 로부터
#  COCO panoptic segmentation 어노테이션형식(json & png 둘다필요)으로 변환가능함!!
#
########################################################################################################################


# # i. COCO object detection 형식.
# {
#     # 중요만.
#     "images": [image],
#     "annotations": [annotation], 
#     "categories": [category]
# }

# image{
#     # 중요만.
#     "id": int, 
#     "width": int, ###################
#     "height": int, ###################
#     "file_name": str, 
# }

# annotation{
#     "id": int, 
#     "image_id": int, 
#     "category_id": int, 
#     "segmentation": RLE or [polygon], ###################
#     "area": float,
#     "bbox": [x,y,width,height], 
#     "iscrowd": 0 or 1,
# }

# category{
#     "id": int, 
#     "name": str, ###################
#     "supercategory": str,
# }

import json, os

# i. 불러올 어노json파일 경로. "path/to/ coco formatted obj det annotation json file for loading"
# i.21.3.11.11:02) 지금은 from_cocoannotator_for_panopticSegJ.json 이런식으로 json파일명 바꿔줬음. 
#  제대로 어노테이션 다수 해준뒤에 확실히 이름 정해서 위 경로 수정할것.
# i.21.3.11.12:43) 기존의 convertTestJ 에서 panopticSeg_dentPanoJ 로 폴더명 바꿨음. 뭐 사실 요 json파일의 위치는 어디에있든 상관없지만 일단 여기에 두는걸로.
COCOFORM_OBJ_DET_ANNOJSON_LOADPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ\from_cocoannotator_for_panopticSegJ.json" 

# i. 저장할 폴더의 경로. "path/to/ dir of cityscapes formatted ~~polygons.json file for saving"
# i.21.3.11.10:57) train폴더 만들어줘서 경로 수정함. 지금 일단 train 폴더만 해줫음. val 폴더도 만들어주면 거기에다가도 ~~polygons.json 들 저장해줘야함!
# i.21.3.11.12:43) 기존의 convertTestJ 에서 panopticSeg_dentPanoJ 로 폴더명 바꿨고, 그안에 gt 및 inputOriPano 두가지 폴더 다시 만들어줬음.
CITYSCAPESFORM_POLYGONS_JSON_SAVEDIRPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ\gt\train" 


# COCO object detection 형식의 어노json파일을 읽어들임.
try:
    with open(COCOFORM_OBJ_DET_ANNOJSON_LOADPATH_J) as f:
        coco_obj_det_anno = json.load(f)
except FileNotFoundError:
    print(f"j) file not found!!: {COCOFORM_OBJ_DET_ANNOJSON_LOADPATH_J}")


# catId to catName
catId2catName = {cat["id"]: cat["name"] for cat in coco_obj_det_anno["categories"]}

# imgId to cityscapes objects
imgId2csObjects = {}
for annotation in coco_obj_det_anno["annotations"]:
    # 1 coco annotation to (maybe multiple) cityscapes polygon(s). (coco 에선 한 annotation 에 polygon 이 여러개잇을수잇음)
    for oneCocoPolygon in annotation["segmentation"]:
        csPolygon = [[x,y] for x,y in zip(oneCocoPolygon[0::2], oneCocoPolygon[1::2])]
        csObject = {
            "label": catId2catName[annotation["category_id"]],
            "polygon": csPolygon,
        }
        if annotation["image_id"] not in imgId2csObjects:
            imgId2csObjects[annotation["image_id"]] = [csObject]
        else:
            imgId2csObjects[annotation["image_id"]].append(csObject)

# 한 이미지당 하나의 (cityscapes의)~~polygons.json 파일을 만듦.
for imgDict in coco_obj_det_anno["images"]:
    cs_polygonsJson_dict = {
        "imgHeight": imgDict["height"],
        "imgWidth": imgDict["width"],
        "objects": imgId2csObjects[imgDict["id"]]
    }
    # i. make json file with cs_polygonsJson_dict.
    savePathJ = os.path.join(CITYSCAPESFORM_POLYGONS_JSON_SAVEDIRPATH_J, \
        os.path.splitext(imgDict["file_name"])[0] + '_polygons.json')
    with open(savePathJ, 'w') as f:
        json.dump(cs_polygonsJson_dict, f)


print(f'j) finished converting.\n\
  {len(coco_obj_det_anno["images"])} images were converted from [COCO obj det format] to [cityscapes ~~polygons.json]')




# # cityscapes 의 ~~polygons.json 형식은 아래와 같음 (한 이미지당 한 json).
# {
#     "imgHeight": 1024, 
#     "imgWidth": 2048, 
#     "objects": [
#         {
#             "label": "sky", 
#             "polygon": [
#                 [
#                     704, 
#                     191
#                 ], 
#                 [
#                     1044, 
#                     404
#                 ], 
#                 [
#                     1293, 
#                     128
#                 ], 
#                 [
#                     1320, 
#                     0
#                 ], 
#                 [
#                     678, 
#                     0
#                 ], 
#                 [
#                     701, 
#                     190
#                 ]
#             ]
#         }, 
#         {
#             "label": "road", 
#             "polygon": [
#                 [
#                     1145, 
#                     391
#                 ], 
#                 [
#                     13, 
#                     467
#                 ], 
#                 [
#                     0, 
#                     466
#                 ], 
#                 [
#                     0, 
#                     1024
#                 ], 
#                 [
#                     2048, 
#                     1024
#                 ], 
#                 [
#                     2048, 
#                     446
#                 ]
#             ]
#         }, 

