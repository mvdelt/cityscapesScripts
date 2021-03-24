

########################################################################################################################
#
# i.21.3.5.19:56) 
#  요약: COCO object detection 형식 -> cityscapes 의 ~~polygons.json 의 형식 으로 형식바꿔주는 코드. 
#  어떤 특정 어노테이션툴에서, 세그멘테이션 어노테이션한걸
#  COCO object detection 형식으로 내뱉었다고 했을때, 
#  그걸 cityscapes 의 ~~polygons.json 의 형식으로 바꿔주기 위한 코드임.
#  ~~polygons.json 만 있으면, cityscapesscripts 의 코드들을 사용해서 
#  원하는형식의 어노테이션png파일들 생성가능하고(cityscapes 에서 어노테이션png 파일들 종류가 몇개 있지),
#  createPanopticImgs.py 이용해서 ~~instanceIds.png 로부터
#  COCO panoptic segmentation 어노테이션형식(json & png 둘다필요)으로 변환가능함!!
#
# i.21.3.14.23:32) 참고로, 여기 내가 써놓은 'COCO object detection 형식' 이라는것은, 
#  정확히말하면 object detection 이 아니고 instance segmentation 임.
#  (COCO 에서는 Object Detection 태스크가 사실 instance segmentation 태스크임. segmentation 까지 다 해줘야하는 태스크임.)
#  TODO: obj det 라고 표기해놓은것들 ins seg 등으로 표기 바꾸는게 나으려나? 나중에 봤을때 헷갈리지 않도록?
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
# COCOFORM_OBJ_DET_ANNOJSON_LOADPATH_J = r"C:\Users\starriet\Downloads\panopticSeg_dentPanoJ\from_cocoannotator_for_panopticSegJ.json" 
# i.21.3.14.23:48) 코랩컴에서의 경로로 변경.
# i.21.3.24.18:24) json파일명에 _forTrain 붙여줌. val 위한 어노json 도 추가하면서, 구분위해.
COCOFORM_OBJ_DET_ANNOJSON_FORTRAIN_LOADPATH_J      = "/content/datasetsJ/panopticSeg_dentPanoJ/from_cocoannotator_for_panopticSegJ_forTrain.json"

# i.21.3.24.18:24) val 위한 어노json 도 추가.
COCOFORM_OBJ_DET_ANNOJSON_FORVAL_LOADPATH_J        = "/content/datasetsJ/panopticSeg_dentPanoJ/from_cocoannotator_for_panopticSegJ_forVal.json"




# i. 저장할 폴더의 경로. "path/to/ dir of cityscapes formatted ~~polygons.json file for saving"
# i.21.3.11.10:57) train폴더 만들어줘서 경로 수정함. 지금 일단 train 폴더만 해줫음. val 폴더도 만들어주면 거기에다가도 ~~polygons.json 들 저장해줘야함!
# i.21.3.11.12:43) 기존의 convertTestJ 에서 panopticSeg_dentPanoJ 로 폴더명 바꿨고, 그안에 gt 및 inputOriPano 두가지 폴더 다시 만들어줬음.
# i.21.3.14.23:48) 코랩컴에서의 경로로 변경.
#  지금 train 폴더에 대해서만 하드코드해놧는데, train 뿐 아니라 val (또는 나아가서 test) 에 대해서도 해줄것. 
CITYSCAPESFORM_POLYGONSJSON_FORTRAIN_SAVEDIRPATH_J = "/content/datasetsJ/panopticSeg_dentPanoJ/gt/train" 

# i.21.3.24.18:28) val 추가.
CITYSCAPESFORM_POLYGONSJSON_FORVAL_SAVEDIRPATH_J   = "/content/datasetsJ/panopticSeg_dentPanoJ/gt/val" 



# i.21.3.24.18:30) val 도 추가했기때문에, for loop 이용해서 train 과 val 에 대해 똑같은작업 반복.
for setName, json_loadpath, polygonsjson_savedirpath in \
    [("train", COCOFORM_OBJ_DET_ANNOJSON_FORTRAIN_LOADPATH_J, CITYSCAPESFORM_POLYGONSJSON_FORTRAIN_SAVEDIRPATH_J), \
     ("val",   COCOFORM_OBJ_DET_ANNOJSON_FORVAL_LOADPATH_J,   CITYSCAPESFORM_POLYGONSJSON_FORVAL_SAVEDIRPATH_J)]:

    print(f'j) for \"{setName}\" set, converting \"시초\"어노json to ~~polygons.json...')

    # COCO object detection 형식의 어노json파일을 읽어들임.
    try:
        with open(json_loadpath) as f:
            coco_obj_det_anno = json.load(f)
    except FileNotFoundError:
        print(f"j) file not found!!: {json_loadpath}")

    # catId to catName
    catId2catName = {cat["id"]: cat["name"] for cat in coco_obj_det_anno["categories"]}

    # i.21.3.17.9:21) 원래필요없어야하는데, 내가사용한어노테이션툴인 'coco-annotator' 의 버그때문에 작성하는부분.
    #  (무슨버그냐면: Rt Lt 구분 없애주려고 coco-annotator 에서 기존 클래스들중 Rt Lt 구분되는 클래스들(sinus, canal) 없애고 
    #  Rt Lt 통합한 클래스들 다시 만들어줬는데, 기존에 어노테이션해둔 파노(겨우2장이긴함)의 기존 Rt Lt 구분되는 클래스들에대한 어노테이션정보가
    #  삭제되지 않고 그대로 남아있는 버그.)
    #  사실 버그해결만 위해서는 뭐 이것도 필요없는데, 버그해결상황좀 확인하기위함임.
    imgId2imgFileName = {img["id"]: img["file_name"] for img in coco_obj_det_anno["images"]}
    bugCatImgFileName2bugCatIds = {} # ex: {'imp2_1.jpg':[36,37], ...}

    # imgId to cityscapes objects
    imgId2csObjects = {}
    for annotation in coco_obj_det_anno["annotations"]:

        # i.21.3.17.9:21) 원래필요없어야하는데, 내가사용한어노테이션툴인 'coco-annotator' 의 버그때문에 작성하는부분.
        #  catId2catName 에 없는 annotation 이면,
        #  imgId2imgFileName 이용해서 어떤 이미지에서 어떤 카테고리id 가 없었던건지 기록하고(이 기록은 그냥 내가 확인해보려는 용도임),
        #  이 annotation 은 패스함 (continue).
        if not annotation["category_id"] in catId2catName:
            bugCatImgFileName = imgId2imgFileName[annotation["image_id"]]
            if bugCatImgFileName in bugCatImgFileName2bugCatIds:
                bugCatImgFileName2bugCatIds[bugCatImgFileName].append(annotation["category_id"])
            else:
                bugCatImgFileName2bugCatIds[bugCatImgFileName] = [annotation["category_id"]]
            continue

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


    # i.21.3.17.10:08) 버그상황 출력.
    print(f'j) 버그있는 이미지 및 버그카테고리들: {bugCatImgFileName2bugCatIds}')


    # 한 이미지당 하나의 (cityscapes의)~~polygons.json 파일을 만듦.
    for imgDict in coco_obj_det_anno["images"]:
        cs_polygonsJson_dict = {
            "imgHeight": imgDict["height"],
            "imgWidth": imgDict["width"],
            "objects": imgId2csObjects[imgDict["id"]]
        }
        # i. make json file with cs_polygonsJson_dict.  # i.21.3.14.23:53) 참고로, 나중에혹시까먹을까봐 적어두는데, cs 라는건 cityscapes 의 줄임말임.
        savePathJ = os.path.join(polygonsjson_savedirpath, \
            os.path.splitext(imgDict["file_name"])[0] + '_polygons.json')
        with open(savePathJ, 'w') as f:
            json.dump(cs_polygonsJson_dict, f)


    print(f'j) finished converting.\n\
    {len(coco_obj_det_anno["images"])} images were converted from [COCO obj det format] to [cityscapes ~~polygons.json]')
    print('-----------------------------------------------')




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

