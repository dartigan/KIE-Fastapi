# conda activate lp2
# curl -X POST "http://127.0.0.1:8000/model_pred/" -F "file=@E:/deepdoctection/ddtp/AWS/imgs/1.PNG"

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import base64
import os
import io
import numpy as np
import cv2
import tempfile
import layoutparser as lp
import random
import os

app = FastAPI()

def convertkey2labels(patches_dict):
    new_dict = {}
    custom_label_map = {0: 'amount',
                        1: 'column',
                        2: 'company_name',
                        3: 'cu_date',
                        4: 'cu_serial_no',
                        5: 'invoice_date',
                        6: 'invoice_no',
                        7: 'po_no',
                        8: 'table',
                        9: 'text'}

    for i in patches_dict.keys():
        new_dict[custom_label_map[i]] = patches_dict[i]

    return new_dict

# curl -X POST "http://127.0.0.1:8000/model_pred/" -F "file=@"E:/deepdoctection/ddtp/AWS/imgs/1.png"



def populate_dict(layout_predicted):
  patches_dict = dict()

  for patch in layout_predicted:
    score = str(patch.score)[:4]
    label = patch.type
    bbox = [np.round(patch.block.x_1,3), np.round(patch.block.y_1,3),
            np.round(patch.block.x_2,3), np.round(patch.block.y_2,3)]

    if (label not in patches_dict) or (len(patches_dict) == 0):
      patches_dict[label] = {'bbox':[bbox], 'score':[str(score)]}
    else :
      patches_dict[label]['bbox'].append(bbox)
      patches_dict[label]['score'].append(score)

  return patches_dict

def annotate_img_cv2(patches_dict, img):
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (255, 165, 0), 
                  (128, 0, 128), (0, 128, 128), (165, 42, 42), 
                  (255, 192, 203)]
    
    color_cnt = 0
    for label_name in patches_dict:
        color = color_list[color_cnt % len(color_list)]
        color_cnt += 1
        for elems in range(len(patches_dict[label_name]['bbox'])):
            score = str(patches_dict[label_name]['score'][elems])  # Ensure score is converted to string
            txt = label_name + "/" + score
            bbox_elems = patches_dict[label_name]['bbox'][elems]
            
            # Convert bbox to integers
            bbox_elems = [int(coord) for coord in bbox_elems]

            # Draw rectangle and text on the image
            cv2.rectangle(img, (bbox_elems[0], bbox_elems[1]), (bbox_elems[2], bbox_elems[3]), color, 2)
            cv2.rectangle(img, (bbox_elems[0], bbox_elems[1] - 20), (bbox_elems[0] + len(txt) * 10, bbox_elems[1]), color, -1)
            cv2.putText(img, txt, (bbox_elems[0] + 5, bbox_elems[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

def detect_anno_areas_lp(image):

    # initializing the weights
    model_config = "E:/deepdoctection/ddtp/AWS/config/config.yaml"
    model_weights = "E:/deepdoctection/ddtp/AWS/config/model_0015999.pth"


    model = lp.Detectron2LayoutModel(model_config,model_weights,
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
    layout_predicted = model.detect(image)

    patches_dict = populate_dict(layout_predicted)

    print()
    # print(patches_dict)
    label_bbox_dict = convertkey2labels(patches_dict)
    # print(label_bbox_dict)

    # removing table and column values
    label_bbox_dict_nocoltab = {i:v for i,v in label_bbox_dict.items() if i not in ['table', 'column', 'text']}
    # pop table and column
    annotated_image = annotate_img_cv2(label_bbox_dict_nocoltab, image)

    return annotated_image, label_bbox_dict

@app.get("/")
def home():
    return {"message" : "Welcome !"}


@app.post("/model_pred/")
async def model_pred(file : UploadFile):
    print("Inside model pred")
    contents = await file.read()
    # filename = file.filename

    # Reading image using cv2
    print("DECODING IMAGE")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    print("MODEL PREDICTION")
    annotated_image, layout_predicted = detect_anno_areas_lp(image)
    print("ANNOTATED IMAGE !!")
    #close
    # cv2.imshow("Annotated_img", annotated_image)
    # cv2.waitKey(0)

    _, encoded_img = cv2.imencode('.PNG' , annotated_image)
    print("ENCODING ANNOTATED IMAGE")
    encoded_img = base64.b64encode(encoded_img)

    return{
        'regions_identified': layout_predicted,
        'encoded_img': encoded_img,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

