import multiprocessing
import numpy as np
import cv2
import random
from collections import deque
import subprocess
import re
import time
import os
from darknet import performDetect


IMGPATH='./img/'
EXT = 'jpg'
#NUMPROCESSES = 2
NUMPROCESSES = 1
MISSLIMIT = 5
THRESHOLD=0.1

#bboxes=[ [198,214,34,81] ]   # - groundtruth
#bboxes=[ [319,214,59,126] ] #yolo detected object-1
#bboxes=[ [182,214,55,112] ] #yolo detected object-2
#bboxes=[ [423,169,36,98], [198,214,34,81], [319,214,59,126] ] #yolo detected object-2
#firstframe=[1,1]
bboxes = []
firstframe=[]
cols=[]
yolopaths = []

curid=1
procid = dict()
misscount = dict()

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
      return 0
    if (a[2]*a[3]>b[2]*b[3]):
        den=float(a[2]*a[3])
    else:
        den=float(b[2]*b[3])
    return(float(w*h/den))

def transformBB(bbox):
    xCoord = int(bbox[0] - bbox[2]/2)
    yCoord = int(bbox[1] - bbox[3]/2)
    width = int(bbox[2])
    height = int(bbox[3])
    new_bb = [xCoord, yCoord, width, height]
    return new_bb

def addbox(bbox,start_frame, freeprocesses, runningprocesses):
    global curid
    p=freeprocesses.popleft()
    procid[p]=curid
    curid+=1
    misscount[p]=0
    p.stdin.write('{} {} false {} {} {} {} {}\n'.format(IMGPATH, EXT, bbox[0],bbox[1], bbox[2], bbox[3], start_frame))
    runningprocesses.append(p)
    print("new process with ID = {} being tracked at frame {} with coordinates {} {} {} {}".format(procid[p], start_frame, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

def kill(index, runningprocesses, cols, global_frame):
    if index>=len(runningprocesses):
        print("killing non-existant process at index {}".format(index))
        return
    print('killing process with ID {}'.format(procid[runningprocesses[index]]))
    print(global_frame)
    procid.pop(runningprocesses[index])
    runningprocesses.pop(index)
    cols.pop(index)

def killprocesses(runningprocesses, cols, removelist, global_frame):
    for i in reversed(removelist):
        kill(i, runningprocesses, cols, global_frame)


def drawrectangles(img_location, x, cols, confidence_score, frame):
    print(img_location)
    img=cv2.imread(img_location,cv2.IMREAD_COLOR)
    res = img_location[0:-9] + '/result/' + img_location[-8:]
    for i in range(len(x)):
        #if confidence_score[i] < THRESHOLD:
        #    continue
        topleft= x[i][0],x[i][1]
        bottomright= x[i][0]+x[i][2],x[i][1]+x[i][3]        
        cv2.rectangle(img,topleft,bottomright,cols[i],2)
        cv2.putText(img,'{}'.format(confidence_score[i]), (x[i][0], x[i][1]-30), 0, 0.8, (0, 255, 0), 2, 2)
    cv2.imwrite(res,img)
    cv2.imshow('image',img)
    cv2.waitKey(50)
    
if __name__=='__main__':
    global IMGPATH,yolopaths
    freeprocesses=deque()
    runningprocesses=[]
    r = lambda: random.randint(0,255)
    paths = []
    filename = IMGPATH
    prevcoords = []
    for path in os.listdir(filename):
        if(path[-3:]=='jpg'):
            paths.append(os.path.abspath(os.path.join(filename,path)))
    
    for i in range(NUMPROCESSES):
        p=subprocess.Popen('python a.py ',shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        freeprocesses.append(p)
    
    ff_bbox = performDetect(1,paths[0])
    yolopaths.append(performDetect(0,paths[0]))

    for i in range(NUMPROCESSES):
        bboxes.append(transformBB(ff_bbox[i][2]))
        firstframe.append(1)

    for i, frame in enumerate(firstframe):
        if frame == 1:
            cols.append([255, 255, 255])
            addbox(bboxes[i], firstframe[i], freeprocesses, runningprocesses)
    
    for i in range(1,len(paths)):
        yolopaths.append(performDetect(0,paths[i]))

    try:
        global_frame = 1
        count = 1
        while True:
            coords = []
            confidence_score = []
            remove_list = []
            for i, conn in enumerate(runningprocesses):
                msg = conn.stdout.readline()
                if re.match(r'^-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+', msg) is None:
                    #print(global_frame)
                    print ('got wrong input {}'.format(msg)) 
                    remove_list.append(i)
                    continue    

                msg=[ float (z)  for z in msg.split(' ')]
                
                if msg[-1] < THRESHOLD:
                    misscount[conn]+=1
                    if misscount[conn]>=MISSLIMIT:
                        
                        del misscount[conn]
                        remove_list.append(i)
                        
                        print('miss limit reached')
                        killprocesses(runningprocesses, cols, remove_list, global_frame)
                        
                        print("Opened new process")
                        p=subprocess.Popen('python a.py ',shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
                        freeprocesses.append(p)
                        cols.append([255, 255, 0])
                        
                        new_bb = performDetect(1,paths[global_frame-1])
                        for r in range(len(new_bb)):
                            print(new_bb[r])
                            print("\n")
                        prevcoords.pop(i)
                        print(prevcoords)
                        for k in range(0,len(new_bb)):
                            newcoords = new_bb[k][2]
                            counter = 0
                            for l in range(0,len(prevcoords)):
                                if intersection(prevcoords[l],transformBB(new_bb[k][2])) > 0.85:
                                    continue
                                else:
                                    #newcoords = new_bb[k][2]
                                    #print(intersection(prevcoords[l],transformBB(new_bb[k][2])))
                                    counter = counter + 1
                            if counter == len(prevcoords):
                                break
                        print(newcoords)
                        newcoords = transformBB(newcoords)
                        addbox(newcoords, global_frame-1, freeprocesses, runningprocesses)
                        remove_list=[]

                        listprocesses = list(enumerate(runningprocesses))
                        nextconn = listprocesses[i][1]
                        newmsg = nextconn.stdout.readline()
                        newmsg=[ float (z)  for z in newmsg.split(' ')]
                        msg = newmsg
                        confidence_score.append(float(newmsg[-1]))
                        newmsg=[int(m) for m in newmsg]
                        coords.append(newmsg)
                        continue
                else:
                    if misscount[conn] >= 1:
                        misscount[conn]-=1
                
                confidence_score.append(float(msg[-1]))
                msg=msg[:4]
                msg=[int(m) for m in msg]
                coords.append(msg)
            prevcoords = coords
            print(str(msg) + '-' + str(global_frame))
            killprocesses(runningprocesses, cols, remove_list, global_frame)
            if not runningprocesses:
                exit(0)
            drawrectangles(paths[global_frame-1], coords, cols, confidence_score, global_frame)
            global_frame+=1

    except KeyboardInterrupt:
        killprocesses(runningprocesses, cols, [i for i in range(len(runningprocesses))])    


    
