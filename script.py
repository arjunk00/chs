import cv2
import tensorflow as tf
import os
import time

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
width = 256
height = 256
dim = (width, height)

cam = cv2.VideoCapture(0)
while True:
    result, image = cam.read()
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # resized_gray=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    img = tf.keras.utils.img_to_array(resized_image)
    print(img.shape)
    final_img = img.reshape(-1, 256, 256, 3)
    tflite_size = os.path.getsize('trashnet.tflite') / 1048576
    tflite_model_path = 'trashnet.tflite'
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
    time.sleep(1)
