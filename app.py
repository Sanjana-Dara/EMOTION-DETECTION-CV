from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import emoji
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #it should load new image every time

@app.route('/') #created a route for index.html
def index():
	return render_template('index.html')

@app.route('/team_members')
def team_members():
    return render_template('aboutus.html')


@app.route('/predict', methods=['GET', 'POST']) #created a route for predict.html
def predict():
	image = request.files['select_file']

	image.save('static/file.jpg')

	image = cv2.imread('static/file.jpg')


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #create a cascade

	faces = cascade.detectMultiScale(gray, 1.1, 3) #detect faces
	cropped = []
	prediction=[]
	predicted=[]
	val_max=[]
	images=[]
	final_pred=[]
	send=list()
	i=0
	temp = 0
	model = load_model('ED_model.h5')  #load the model
	label_map = ['Anger', 'Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 2
	color = (0, 0, 255)
	thickness = 3
	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		cropped.append(image[y:y+h, x:x+w])  #to save a cropped image
		try:
			cv2.imwrite('static/cropped'+str(i+1)+'.jpg', cropped[i])
		except:
			pass #in some cases it will not detect face ==>no cropped variable
		org = (x+w, y+h)
		j=str(i+1)
		image = cv2.putText(image, j, org, font, fontScale, color, thickness, cv2.LINE_AA)
		try:
			img = cv2.imread('static/cropped'+str(i+1)+'.jpg', 0)

		except:
			img = cv2.imread('static/file.jpg', 0)

		img = cv2.resize(img, (48,48))#convert to 48 by 48 pixel
		img = img/255.0

		img = img.reshape(1,48,48,1) #reshape the array to 1 by size 48 48

		prediction.append(model.predict(img)) #store the prediction

		#the prediction consist of 6 values in our list so we have to take maximum value
		predicted.append(int(np.argmax(prediction[i])))

		temp = predicted[i]
		val_max.append(max(prediction[i]))

		#final_pred.append(str(i+1)+ " : "+label_map[pred1[i]]+ "\t,  "+ str(round(m1[i][temp]*100,2)))
		final_pred.append(str(round(val_max[i][temp]*100,2)))
		send.append(label_map[predicted[i]])
		i=i+1

	cv2.imwrite('static/after.jpg', image)


	return render_template('predict.html', len = i, data=final_pred, send=send)

if __name__ == "__main__":
	app.run(debug=True)
