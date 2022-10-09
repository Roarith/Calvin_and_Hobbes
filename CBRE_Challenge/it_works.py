import cv2
import numpy as np
from pytesseract import pytesseract

#FOLLOWING IMPORTS AND CODE FOR COMPARISON#
from numpy import round
import json
import sys
import torch

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#END OF COMPARISON MODEL 1#

#MAIN CODE#
pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("cartoon1.png") #image is read in

def grayscale(image): #grayscales the image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image): #uses median blur to remove grain
    return cv2.medianBlur(image, 5)

def thresholding(image): #OTSU
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def sharpen(image): #using a kernel setting applied over the resolution to sharpen
    kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


#different functions to use in combination to modify the image
img = grayscale(img)
img = thresholding(img)
#img = sharpen(img)
#img = noise_removal(img)

words = pytesseract.image_to_string(img) #output string

print(words) #output text read in from image

#BELOW IS THE TEST FOR SIMILARITY#
temp = "...I'M GETTING A RIDE WITH KATIE TO ADAM'S WEDDING HOPING TO SEE BRIAN ON THE WAY! OH THAT'S COOL! ...I CAN'T KEEP LIVING THIS LIE, SO I'M JUST GONNA COME OUT AND SAY IT: I HAVE NO IDEA WHO ANY OF THE PEOPLE YOU KEEP MENTIONING ARE."
newTemp = [temp]
newWords = [words]
    
embedding = model.encode(newTemp + newWords)
cos_sim = util.cos_sim(embedding, embedding)
cos_sim.tolist()
print(cos_sim[0][1]) #decimal is % for similarity/accuracy

