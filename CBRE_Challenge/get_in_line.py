import cv2
import numpy as np
import easyocr

im_1_path = './cartoon1.png'
im_2_path = './cartoon2.PNG'
im_3_path = './cartoon3.PNG'
im_4_path = './cartoon4.PNG'
im_5_path = './cartoon5.PNG'

def recognize_text(img_path):
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)


result = recognize_text(im_1_path)
msg =[]

for i in result:
    msg.append(i[-2])

omsg=''

for i in msg:
    omsg += i + ' '

print(omsg)

#print(result)




