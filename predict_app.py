import base64
import numpy as np
import io
import cv2
from PIL import Image
# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
	global model
	print("Loading Model")
	model = load_model('face_detector_best.h5')
	print(" * Model loaded!")

def face_extractor(image):
	
	faces = face_classifier.detectMultiScale(image, 1.3, 5)

	if faces is ():
	    return None
	for (x,y,w,h) in faces:
	    x-=10
	    y-=10
	    cropped_face = image[y:y+h+50, x:x+w+50]
	    
	return cropped_face

def preprocess_image(image):
	if(image.shape != (250,250,3)):
		print('Image is being resized')
		image = cv2.resize(face_extractor(image),(250,250))
	else:
		print('No need to resize image')
		image = cv2.resize(image,(250,250))

	cv2.imshow('img',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	image = np.array(image, dtype='float32')
	image /= 255
	image = image.reshape(1,250,250,3)
	

	return image

print(" * Loading Keras Model....")
get_model()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route("/predict", methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	image.show()
	# im2 = (np.array(image))[:, :, ::-1].copy() 
	im2 = np.array(image)      # dont convert this to BGR
	
	processed_image = preprocess_image(im2)
	prediction = model.predict(processed_image)
	print(prediction)
	classes = ['Colin Powell', 'George Bush', 'Tony Blair']
	ans = classes[np.argmax(prediction)]
	response = {
		'prediction': str(ans)
	} 
	return jsonify(response)