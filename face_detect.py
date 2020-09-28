import cv2
import numpy as np

input_data_path = './downloads/Steve Jobs/'
save_path = './cutted_jobs/'

cascade_path = '/Users/matsuyamashinji/opt/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

image_count = 400
face_detect_count = 0

for i in range(image_count):
  img = cv2.imread(input_data_path + str(i+1) + '.jpg', cv2.IMREAD_COLOR)
  print(img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)
  if len(face) > 0:
    for rect in face:
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]
      cv2.imwrite(save_path + 'cutted_jobs' + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
      face_detect_count = face_detect_count + 1
    else:
      print('image' + str(i) + ':NoFace')

