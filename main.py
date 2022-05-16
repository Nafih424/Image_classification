import cv2
import joblib
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pandas as pd
import json
import streamlit as st
from PIL import Image, ImageOps
import re



__class_name_to_number = {}
__class_number_to_name = {}



st.write("""
         # Actor  Classification
         """
         )

st.subheader("CLASSIFICATION OF DULQUAR , TOVINO , MAMMOOTTY, MOHANLAL")



img1 = ("./test/dq.jpg")


file = st.file_uploader("Please upload image file", type=["jpg", "png","jpeg"])




st.set_option('deprecation.showfileUploaderEncoding', False)




with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}



def class_number_to_name(class_num):
    return __class_number_to_name[class_num]




with open("./artifacts/saved_model.pkl","rb") as f:
    model = joblib.load(f)


def classify_image(file_path):

    imgs = get_cropped_image_if_2_eyes(file_path)


    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        class_no = model.predict(final)
        print(class_no)
        result.append({
            'class': class_number_to_name(model.predict(final)[0]),
            #'class_probability': np.around(model.predict_proba(final)*100,2).tolist()[0],
            #'class_dictionary': __class_name_to_number
        })
        df = pd.DataFrame(result)
       
    return df

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H





def get_cropped_image_if_2_eyes(image_path):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

#df = classify_image(img1)
#st.write(df)

def prediction(x):
	pred = classify_image(x)
	st.write(pred)
	print(
    "This image most likely belongs to {}"
    	)
if file is None:
	st.text("please upload the image")
else:
	fileName = file.name
	path = ("./test/"+ fileName)
	#path = ('"' + path + '"')
	#path = re.sub("[']","",path)
	print(path)
	prediction(path)