# -*- coding: utf-8 -*-
import cv2
import detect_face
import tensorflow as tf
import numpy as np
import subprocess
import os
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

class FFmpegVideoCapture:
    # mode=gray,yuv420p,rgb24,bgr24
    def __init__(self,source,width,height,mode="gray",start_seconds=0,duration=0,verbose=False):

        x = ['ffmpeg']
        if start_seconds > 0:
            #[-][HH:]MM:SS[.m...]
            #[-]S+[.m...]
            x.append("-accurate_seek")
            x.append("-ss")
            x.append("%f" % start_seconds)
        if duration > 0:
            x.append("-t")
            x.append("%f" % duration)
        x.extend(['-i', source,"-f","rawvideo", "-pix_fmt" ,mode,"-"])
        self.nulldev = open(os.devnull,"w") if not verbose else None
        self.ffmpeg = subprocess.Popen(x, stdout = subprocess.PIPE, stderr=subprocess.STDERR if verbose else self.nulldev)
        self.width = width
        self.height = height
        self.mode = mode
        if self.mode == "gray":
            self.fs = width*height
        elif self.mode == "yuv420p":
            self.fs = width*height*6/4
        elif self.mode == "rgb24" or self.mode == "bgr24":
            self.fs = width*height*3
        self.output = self.ffmpeg.stdout
    def read(self):
        if self.ffmpeg.poll():
            return False,None
        x = self.output.read(self.fs)
        if x == "":
            return False,None
        if self.mode == "gray":
            return True,np.frombuffer(x,dtype=np.uint8).reshape((self.height,self.width))
        elif self.mode == "yuv420p":
            # Y fullsize
            # U w/2 h/2
            # V w/2 h/2
            k = self.width*self.height
            return True,(np.frombuffer(x[0:k],dtype=np.uint8).reshape((self.height,self.width)),
                np.frombuffer(x[k:k+(k/4)],dtype=np.uint8).reshape((self.height/2,self.width/2)),
                np.frombuffer(x[k+(k/4):],dtype=np.uint8).reshape((self.height/2,self.width/2))
                    )
        elif self.mode == "bgr24" or self.mode == "rgb24":
            return True,(np.frombuffer(x,dtype=np.uint8).reshape((self.height,self.width,3)))
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    stream = "rtsp://admin:mc123456@172.16.63.200:554/h264/ch1/sub"
    cap1 = FFmpegVideoCapture(stream,640,480,"bgr24")
    #print cap1.isOpened()
    c=0

    while True:
        ret, frame = cap1.read()
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
    #cap.release()
    cv2.destroyAllWindows()

