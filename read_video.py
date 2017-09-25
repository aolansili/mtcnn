# -*- coding: utf-8 -*-
import cv2
import numpy as np
import detect_face
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
frame_interval=3 # frame intervals

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

image_path = '3.jpg'

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model/')

#image_path = 'timg.jpeg'
#img = misc.imread(image_path)

cap = cv2.VideoCapture(0)
c=0

while True:
    ret, frame = cap.read(0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if gray.ndim == 2:
        img = to_rgb(gray)

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]  # 人脸数目
    print('找到人脸数目为：{}'.format(nrof_faces))

    #print(bounding_boxes)
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 1)

    cv2.imshow("video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

