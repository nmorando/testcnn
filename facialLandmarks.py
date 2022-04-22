
from imutils import face_utils
import dlib
import cv2
from PIL import Image
import numpy as np
import sys
import imutils
import resource
import os
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from datetime import datetime
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
classifier = load_model('blinking_cnn_model.h5')

#provare con stesso dataset
# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "../shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)
counter = 100
fps = FPS().start()

while True:
    # Getting out image by webcam
    (grabbed, gray) = cap.read()
    gray = imutils.resize(gray, width=400)
    cap.set(1, 30)

    # Converting the image to gray scale
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    cv2.imshow("viso", gray)


    # Get faces into webcam's image
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):

        shapex = predictor(gray, rect)
        shapex = face_utils.shape_to_np(shapex)
 
    x1 = shapex[36][0]
    y1 = shapex[19][1]
    x2 = shapex[29][0]#39
    y2 = shapex[29][1]#41

 




    img = gray[y1:y2, x1:x2]

    
   # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


    dim = (86, 86)
    img = cv2.resize(img, dim)

    cv2.imwrite('a.jpg',img)
#       counter= counter +1

    cv2.imshow("OutputAdaptiveThreshold", img)
    test_image =image.load_img('a.jpg',target_size =(86,86,3))
    test_image =image.img_to_array(test_image)
    test_image = test_image/255
    test_image =np.expand_dims(test_image, axis =0)
    result = classifier.predict_proba(test_image)
#       if(counter == 100):
#          print("seconda parte")
    counterM = 1
    counterOpe = 0
    counterClos = 0
    
    prediction = 'open'
    if result[0][0] >= 0.5:
       prediction = 'open'
       print("open")
    else:
       prediction = 'closed'
       print("closed")
        
        #   print(result[0][0])
   
    now = datetime.now()

    cv2.imwrite(prediction+str(now)+'.jpg',img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    window_name = 'Image'
      
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
      
    # org
    org = (50, 50)
      
    # fontScale
    fontScale = 1
       
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2
       
    # Using cv2.putText() method
    path = r'white.png'
        
    # Reading an image in default mode
    imagexx = cv2.imread(path)
    imagexx = cv2.putText(imagexx, prediction, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
       
    # Displaying the image
    cv2.imshow(window_name, imagexx)
    fps.update()



cv2.destroyAllWindows()
cap.release()
