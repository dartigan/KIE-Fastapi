import requests
import json
import os
import base64
import cv2
import numpy as np
import tempfile
import pdfplumber
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
Paddle = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)


# filepath = "E:/deepdoctection/ddtp/AWS/imgs/3.png"
filepath  = "E:/deepdoctection/ddtp/AWS/pdfs/SI236012376_Evergreen Fresh Dist Ltd.pdf"
# filepath = "E:/deepdoctection/ddtp/AWS/readme.txt"

model_pred_url = "http://127.0.0.1:8000/model_pred/"


def read_data(format, filepath,regions_identified):
    kie_dict = dict()
    exclude_keys = ['table', 'column', 'text']
    kie_keys = [i for i in regions_identified.keys() if i not in exclude_keys]
    
    if format == 'pdf' :
        pdf_doc = pdfplumber.open(filepath)
        first_page = pdf_doc.pages[0]
        for keys in kie_keys:
            for elem in regions_identified[keys]['bbox']:
                cropped_page = first_page.within_bbox(elem)
                text = cropped_page.extract_text()
                if (keys not in kie_dict.keys()): # or (len(kie_dict.keys()) == 0):
                    kie_dict[keys] = text
                else:
                    kie_dict[keys].append(text)
    
    elif format == 'image':
        print(filepath)
        image = cv2.imread(filepath)
        for keys in kie_keys:
            for elem in regions_identified[keys]['bbox']:
                cropped_area = image[int(elem[1]):int(elem[3]) ,int(elem[0]):int(elem[2])]
                text = Paddle.ocr(cropped_area, cls=False, det=True, rec=True)
                try :
                    for idx in range(len(text)):
                        res = text[idx]
                        for line in res:
                            data = line[1][0]

                    if (keys not in kie_dict.keys()): # or (len(kie_dict.keys()) == 0):
                        kie_dict[keys] = data
                    else:
                        kie_dict[keys].append(data)
                except BaseException as e:
                    kie_dict = {}
    return kie_dict


def post_request(filepath, model_pred_url):
    headers = {
        'content-type': 'application/json',
        'img_filepath': filepath,
        'filename': filepath.split("/")[-1]
        }

    with open(filepath, 'rb') as f:
        response = requests.post(model_pred_url, files={'file': f.read()}) #, headers=headers)
        response_json = json.loads(response.text)
        print()
        # print('Image :',response_json['encoded_img'])
        # print('Regions :',response_json['regions_identified'])

        img = base64.b64decode(response_json['encoded_img'])
        image = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)

        # cv2.imshow("Annotated Image",image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return image, response_json['regions_identified']


def show_image(image):
    cv2.imshow("Annotated Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if filepath.lower().endswith(".pdf"):
    format = 'pdf'
    # convert pdf to image
    pdf_doc = pdfplumber.open(filepath)

    with tempfile.TemporaryDirectory() as path:
        images_from_path = convert_from_path(filepath, output_folder=path, fmt='png')
        for img_page in range(len(images_from_path)):
            dims =[int(pdf_doc.pages[img_page].width), int(pdf_doc.pages[img_page].height)]
            image = images_from_path[img_page].resize((dims[0], dims[1]))

            temp_image_path = os.path.join(path, f"{filepath[:-4]}_{img_page}.png")
            image.save(temp_image_path)
            ann_image, regions_identified = post_request(temp_image_path, model_pred_url)
            os.remove(temp_image_path)
            show_image(ann_image)

            kie_dict = read_data(format, filepath,regions_identified)
            print(kie_dict)
    # call data to read

elif (filepath.lower().endswith(".png") or filepath.lower().endswith(".jpg") or filepath.lower().endswith(".jpeg")):
    format = 'image'
    ann_image, regions_identified = post_request(filepath, model_pred_url)
    show_image(ann_image)
    kie_dict = read_data(format, filepath,regions_identified)
    print(kie_dict)
else :
    print(f'Error : "{filepath[-3:]}" extension for file "{filepath.split("/")[-1]}" not supported !!')