# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
count=0

prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
image1 = 'Left8bits/img_00050.jpg'
image2 = 'Left8bits/img_00150.jpg'
image3 = 'Left8bits/img_00350.jpg'

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image50 = cv2.imread(image1)
imageOriginal50 = cv2.imread(image1)
#equ = cv2.equalizeHist(image)
img_yuv50 = cv2.cvtColor(image50, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv50[:,:,0] = cv2.equalizeHist(img_yuv50[:,:,0])

# convert the YUV image back to RGB format
equ50 = cv2.cvtColor(img_yuv50, cv2.COLOR_YUV2BGR)

(h, w) = equ50.shape[:2]
print(w,h)
blob = cv2.dnn.blobFromImage(equ50, 0.007843, (640, 480), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	#if confidence > args["confidence"]:
	if confidence > 0.2:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		if format(CLASSES[idx])=="car":
                        count=count+1
                        print("[INFO] {}".format(label))
		cv2.rectangle(equ50, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(equ50, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
# show the output image
cv2.imshow("Output", equ50)
res50 = np.hstack((imageOriginal50,equ50)) #stacking images side-by-side

##########image 150
image150 = cv2.imread(image2)
imageOriginal150 = cv2.imread(image2)
#equ = cv2.equalizeHist(image)
img_yuv150 = cv2.cvtColor(image150, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv150[:,:,0] = cv2.equalizeHist(img_yuv150[:,:,0])

# convert the YUV image back to RGB format
equ150 = cv2.cvtColor(img_yuv150, cv2.COLOR_YUV2BGR)

(h, w) = equ150.shape[:2]
print(w,h)
blob = cv2.dnn.blobFromImage(equ150, 0.007843, (640, 480), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	#if confidence > args["confidence"]:
	if confidence > 0.2:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		if format(CLASSES[idx])=="car":
                        count=count+1
                        print("[INFO] {}".format(label))
		cv2.rectangle(equ150, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(equ150, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
# show the output image
cv2.imshow("Output", equ150)
res150 = np.hstack((imageOriginal150,equ150)) #stacking images side-by-side

res50150 = np.vstack((res50,res150))

################image 350
image350 = cv2.imread(image3)
imageOriginal350 = cv2.imread(image3)
#equ = cv2.equalizeHist(image)
img_yuv350 = cv2.cvtColor(image350, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv350[:,:,0] = cv2.equalizeHist(img_yuv350[:,:,0])

# convert the YUV image back to RGB format
equ350 = cv2.cvtColor(img_yuv350, cv2.COLOR_YUV2BGR)

(h, w) = equ350.shape[:2]
print(w,h)
blob = cv2.dnn.blobFromImage(equ350, 0.007843, (640, 480), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	#if confidence > args["confidence"]:
	if confidence > 0.2:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		if format(CLASSES[idx])=="car":
                        count=count+1
                        print("[INFO] {}".format(label))
		cv2.rectangle(equ350, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(equ350, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
# show the output image
cv2.imshow("Output", equ350)
res350 = np.hstack((imageOriginal350,equ350)) #stacking images side-by-side

res50150350= np.vstack((res50150,res350))

cv2.imwrite('res5015035.jpg',res50150350)
cv2.waitKey(0)
