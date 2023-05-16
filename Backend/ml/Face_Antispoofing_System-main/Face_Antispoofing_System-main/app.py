# An object of Flask class is our WSGI application.
from flask import Flask, request
from flask_cors import CORS
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json


 
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app) 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.


root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

model1 = model_from_json(open("fer.json", "r").read())  
#load weights  
model1.load_weights('fer.h5')  
print("Model1 loaded from disk")






def liveness():
 print(" liveness call2")

 video = cv2.VideoCapture(0)
 print(" liveness call3")

 a=True
 list =[]
 while a:
    try:
        print(len(list))
        if len(list)>=100:
               video.release() 
               cv2.destroyAllWindows()  
               return list
        ret,frame = video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            print(preds)
            if preds> 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)
                list.append(-1)
            else:
                label = 'real'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                (0, 255, 0), 2)
                list.append(1)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
   
 video.release() 
 cv2.destroyAllWindows()






def emotion():
 
 cap=cv2.VideoCapture(0)  
  
 list =[] 
 while True: 
    if len(list)>=100:
              cap.release() 
              cv2.destroyAllWindows()   
              return list
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
    
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
  
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)  
  
  
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = img_to_array(roi_gray) 
       
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model1.predict(img_pixels)  
  
        #find max indexed array  
        max_index = np.argmax(predictions[0])  
        print(max_index)
  
        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'Neutral')  
        emotions = ('fearless', 'fear', 'fear', 'fearless', 'fear', 'fearless', 'fearless')  

        predicted_emotion = emotions[max_index]  
        if(predicted_emotion=='fearless'):
            list.append(1)
            print(predicted_emotion)
        else:
             list.append(-1)
             print(predicted_emotion)
                
  
        cv2.putText(test_img, predicted_emotion, (x,y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.rectangle(test_img, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)  
    resized_img = cv2.resize(test_img, (1000, 700))  
    cv2.imshow('Facial emotion analysis ',test_img)  
  
  
  
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed  
        break  
  
 cap.release()  
 cv2.destroyAllWindows 







@app.route('/liveness',methods=['GET','POST'])
def surajkannujiya():
     if request.method == 'GET':
         print(" liveness call1")
         i= liveness()
         k=sum(i)
         if(k>0):
              return " real"
         else:
             return "spoof"

@app.route('/emotion',methods=['GET','POST'])
def sueaj():
    if request.method=='GET':
        i=emotion()
        k=sum(i)
        if(k>0):
             return " Fearless" 
        else:
            return "Fear"            

  
     
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()