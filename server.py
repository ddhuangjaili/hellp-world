from darknet import darknet
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import codecs
import sys
import signal
import os 

def signal_handler(signum, stack):
    with codecs.open('/root/htsc_detect/args.conn','r','utf-8') as f:
        pid, src_path, vid  = [i.strip('\n') for i in f.readlines()]
    result = detect(src_path, vid, net, meta)
    result = str(result)
    with codecs.open('/root/htsc_detect/result.conn','w','utf-8') as f:
        f.write(result)
    os.kill(int(pid),1)
    
def model_init():
    net = darknet.load_net("/root/htsc_detect/darknet/htsc_test.cfg".encode('utf-8'), "/root/htsc_detect/darknet/htsc.weights".encode('utf-8'), 0) #模型载入
    meta = darknet.load_meta("/root/htsc_detect/darknet/htsc.data".encode('utf-8')) #Processor载入
    return net, meta

def detect(path,detect_vid,net,meta):
    if detect_vid == True:
        cap = cv2.VideoCapture(path)

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(cap.get(5))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        vout_1 = cv2.VideoWriter('output.avi',fourcc,fps,size)
        tracker = cv2.TrackerKCF_create()
        cnt = 0
        tracking = False
        while True:
            print('\r%d'%cnt,end='')
            ret, frame = cap.read()
            if ret != True:
                break

            if tracking == False:
                cv2.imwrite('temp.jpg',frame)
                detect_result = darknet.detect(net, meta, 'temp.jpg'.encode('utf-8')) #调用模型
                for bbox in detect_result:
                    label = bbox[0].decode('utf-8').strip('\r')
                    x_center, y_center, w, h = bbox[2]
                    x1 = int(x_center - w/2)
                    x2 = int(x_center + w/2)
                    y1 = int(y_center - h/2)
                    y2 = int(y_center + h/2)
                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    ok = tracker.init(frame, bbox[2])
                    tracking = True
            if(tracking==True):
                tracking, tracking_box = tracker.update(frame)
                x_center, y_center, w, h = tracking_box
                x1 = int(x_center - w/2)
                x2 = int(x_center + w/2)
                y1 = int(y_center - h/2)
                y2 = int(y_center + h/2)
                p1 = (x1,y1)
                p2 = (x2,y2)
                frame = cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            vout_1.write(frame)
            print(tracking)
            cnt += 1
    else:
        if(os.path.exists(path)):
            img = cv2.imread(path)
            detect_result = darknet.detect(net, meta, path.encode('utf-8'))
            img.shape
            if(detect_result==[]):
                return False
            label = {bbox[0].decode('utf-8').strip('\r') for bbox in detect_result}
            if 'logo' not in label:
                return False
            else:
                if('caption' in label):
                    return True
                else:
                    return False
        elif(path[-3:] not in {'jpg','jpeg','JPG'}):
            return 'Error in ext'
        else:
            return 'Unknown Error'
    
if __name__ == '__main__':
    global net, meta
    net, meta = model_init()
    with codecs.open('pid.conn','w','utf-8') as f:
        f.write(str(os.getpid()))
    while True:
        signal.signal(1,signal_handler)
'''
    if(len(sys.argv)!=3):
        print('Error in number of argument') 
    path = sys.argv[1]
    detect_vid = sys.argv[2]
    hasLogo = detect(path,detect_vid,net,meta)
    print(hasLogo)
'''