import cv2
import tensorflow as tf
import os
import time
# from PIL import Image, ImageOps
import pyttsx3
import numpy as np
#engine=pyttsx3.init()
#engine.setProperty('volume',1.0)

labels = ['paper', 'metal', 'glass', 'cardboard', 'plastic']
width = 224
height = 224
dim = (width, height)

cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    # cv2.imshow('image',image)
    # resized_image = cv2.resize(image, dim, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    # resized_gray=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    # img = tf.keras.utils.img_to_array(resized_image)
    # print(img.shape)
    # final_img = img.reshape(-1, 224, 224, 3)

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    final_img = (image / 127.5) - 1

    tflite_size = os.path.getsize('trashnet.tflite') / 1048576
    tflite_model_path = 'trashnetv2.tflite'
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = final_img
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
    final_prediction = output_data_tflite.argmax()
    pred = labels[final_prediction]
    print(pred)
    engine=pyttsx3.init()
	#engine.setProperty('volume',1.0)
    engine.say(pred)
    engine.runAndWait()
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
    time.sleep(0.3)

camera.release()
cv2.destroyAllWindows()


# from keras.models import load_model  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np
#
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = load_model("keras_Model.h5", compile=False)
#
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
#
# # CAMERA can be 0 or 1 based on default camera of your computer
# camera = cv2.VideoCapture(0)
#
# while True:
#     # Grab the webcamera's image.
#     ret, image = camera.read()
#
#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#
#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)
#
#     # Make the image a numpy array and reshape it to the models input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#
#     # Normalize the image array
#     image = (image / 127.5) - 1
#
#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#
#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#
#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)
#
#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break
#
# camera.release()
# cv2.destroyAllWindows()
