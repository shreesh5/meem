#import statements for Tkinter
import Tkinter as tk, tkFileDialog, tkSimpleDialog, tkMessageBox
from PIL import ImageTk, Image
import time
import os

#import statements for MEEM
import multiprocessing
import numpy as np
import cv2
import random
from collections import deque
import subprocess
import re
from darknet import performDetect

#import statements for YOLO
from skimage import io, draw

#Initialize Tkinter
root = tk.Tk()
root.geometry("1280x780")
#root.configure(background='black')
root.title("GUI")

#global variables for MEEM multi-object tracking
IMGPATH = ''
EXT = ''
NUMPROCESSES = 0
MISSLIMIT = 5
THRESHOLD = 0.1
bboxes = []
firstframe = []
cols = []
curid = 1
procid = dict()
misscount = dict()

#global variables for YOLO bounding boxes
yolo_bb = []
paths = []

#Function to create video
def writeVideo():
	global yolo_bb
	global paths, NUMPROCESSES, IMGPATH, EXT, bboxes, firstframe, curid, cols, procid, misscount
	meem_paths = []
	yolo_paths = []
	meem_res = IMGPATH + '/result-meem'
	yolo_res = IMGPATH + '/result-yolo'
	#vid_path = IMGPATH + '/result.avi'
	for path in os.listdir(meem_res):
		if(path[-3:]=='jpg'):
			meem_paths.append(os.path.abspath(os.path.join(meem_res,path)))
	for path in os.listdir(yolo_res):
		if(path[-3:]=='jpg'):
			yolo_paths.append(os.path.abspath(os.path.join(yolo_res,path)))
	img = cv2.imread(meem_paths[0])
	height, width, layers = img.shape
	size = (640*2, 360)
	out = cv2.VideoWriter('result-test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 12, size)
	for p in range(len(meem_paths)):
		meem = cv2.imread(meem_paths[p],cv2.IMREAD_COLOR)
		meem_resize = cv2.resize(meem,(640,360))
		yolo = cv2.imread(yolo_paths[p],cv2.IMREAD_COLOR)
		yolo_resize = cv2.resize(yolo,(640,360))
		vis = np.concatenate((yolo_resize,meem_resize), axis=1)
		out.write(vis)
	out.release()
	tkMessageBox.showinfo("Video Complete", "The output video has been saved to the root folder")
	paths = []
	yolo_paths = []
	NUMPROCESSES = 0
	IMGPATH = ''
	bboxes = []
	firstframe = []
	curid = 1
	cols = []
	procid = dict()
	misscount = dict()
	
#Function to transform YOLO input for MEEM
def transformBB(bbox):
	xCoord = int(bbox[0] - bbox[2]/2)
	yCoord = int(bbox[1] - bbox[3]/2)
	width = int(bbox[2])
	height = int(bbox[3])
	new_bb = [xCoord, yCoord, width, height]
	return new_bb

#Function to calculate the intersection over union
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
      return 0
    if (a[2]*a[3]>b[2]*b[3]):
        den=float(b[2]*b[3])
    else:
        den=float(a[2]*a[3])
    return(float(w*h/den))

#Function to give input to each MEEM subprocess
def addbox(bbox,start_frame, freeprocesses, runningprocesses):
    global curid, paths
    p=freeprocesses.popleft()
    procid[p]=curid
    curid+=1
    misscount[p]=0
    p.stdin.write('{} {} false {} {} {} {} {}\n'.format(IMGPATH, EXT, bbox[0],bbox[1], bbox[2], bbox[3], start_frame))
    runningprocesses.append(p)
    print("new process with ID = {} being tracked at frame {} with coordinates {} {} {} {}".format(procid[p], start_frame, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

#Function to kill subprocess
def kill(index, runningprocesses, cols):
    if index>=len(runningprocesses):
        print("killing non-existant process at index {}".format(index))
        return
    print('killing process with ID {}'.format(procid[runningprocesses[index]]))
    procid.pop(runningprocesses[index])
    runningprocesses.pop(index)
    cols.pop(index)

#Function to initialize killing of subprocesses
def killprocesses(runningprocesses, cols, removelist):
    for i in reversed(removelist):
        kill(i, runningprocesses, cols)

#Function to save MEEM output
def drawrectangles(img_location, x, cols, confidence_score, frame, runningprocesses):
    img=cv2.imread(img_location,cv2.IMREAD_COLOR)
    indexes = []
    for i in runningprocesses:
    	indexes.append(procid[i])
    for i in range(len(x)):
        #if confidence_score[i] < THRESHOLD:
        #    continue
        topleft= x[i][0],x[i][1]
        bottomright= x[i][0]+x[i][2],x[i][1]+x[i][3]        
        cv2.rectangle(img,topleft,bottomright,cols[i],5)
    	cv2.putText(img,'{}'.format(indexes[i]), (x[i][0], x[i][1]-30), 0, 0.8, (0, 255, 0), 2, 2)
    meem_respath = img_location[0:-9] + '/result-meem/' + img_location[-8:]
    cv2.imwrite(meem_respath,img)
    return  meem_respath

def loopTemp():
	resultpath = []
	filename = IMGPATH + '/result-meem/'
	for path in os.listdir(filename):
		if(path[-3:]=='jpg'):
			resultpath.append(os.path.abspath(os.path.join(filename,path)))
	for x in range(0, len(resultpath)):
		thermostat = ImageTk.PhotoImage(Image.open(resultpath[x]))
		panel.configure(image = thermostat)
		panel1.configure(image = thermostat)
		root.update_idletasks()
		time.sleep(0.75)

#Function to get the directory of images
def getPath():
	global IMGPATH, EXT, paths
	filename = tkFileDialog.askdirectory()
	print(filename)
	IMGPATH = filename
	for path in os.listdir(filename):
		if(path[-3:]=='jpg'):
			paths.append(os.path.abspath(os.path.join(filename,path)))
	print(len(paths))
	if len(paths) == 0:
		lflag = 0
		while lflag == 0:
			print("Directory has no frames. Please enter correct path")
			filename = tkFileDialog.askdirectory()
			print(filename)
			IMGPATH = filename
			for path in os.listdir(filename):
				if(path[-3:]=='jpg'):
					paths.append(os.path.abspath(os.path.join(filename,path)))
			if len(paths) > 0:
				lflag = 1
	print(len(paths))
	if(paths[0][-3:] == 'png' or paths[0][-3:] == 'jpg'):
		EXT = paths[0][-3:]
	elif(paths[0][-4:] == 'jpeg'):
		EXT = paths[0][-4:]
	print(EXT)

#Function to get the number of objects
def getNum():
	global NUMPROCESSES
	n = tkSimpleDialog.askinteger("Number of Objects","Enter the number of objects (1-4)",parent=root,minvalue=1,maxvalue=4)
	print(n)
	NUMPROCESSES = n

#Function to track objects
def startTrack():
	global IMGPATH, NUMPROCESSES, paths, firstframe
	if not (IMGPATH and NUMPROCESSES):
		if not(IMGPATH):
			print("Please specify the directory of images")
		if not(NUMPROCESSES):
			print("Please specify the number of objects to be tracked")
	else:
		ff_bbox = performDetect(1,paths[0])
		for r in range(len(ff_bbox)):
			print(ff_bbox[r])
			print("\n")
		yolo_bb.append(performDetect(0,paths[0]))
        for i in range(NUMPROCESSES):
        	bboxes.append(transformBB(ff_bbox[i][2]))
        	firstframe.append(1)
        print(bboxes)
        print("\n")
        
        freeprocesses = deque()
        runningprocesses = []
        prevcoords = []

        #for i in range(1,len(paths)):
        #	yolo_bb.append(performDetect(0,paths[i]))

        for i in range(NUMPROCESSES):
        	p = subprocess.Popen('python a.py', shell = True, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
        	freeprocesses.append(p)

        for i, frame in enumerate(firstframe):
        	if frame == 1:
        		cols.append([0, 0, 255])
        		addbox(bboxes[i], firstframe[i], freeprocesses, runningprocesses)
        
        count = 1
        try:

        	global_frame = 1
        	flag = 1
        	while flag == 1:
        		coords = []
        		confidence_score = []
        		remove_list = []

        		for i, conn in enumerate(runningprocesses):
        			msg = conn.stdout.readline()
        			if re.match(r'^-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+', msg) is None:
        				print('Received wrong input')
        				for v, vconn in enumerate(runningprocesses):
        					remove_list.append(v)
        				#remove_list.append(i)
        				#continue
        				break
        			msg = [ float(z) for z in msg.split(' ') ]

        			if msg[-1] < THRESHOLD:
        				misscount[conn] += 1
        				if misscount[conn] == 5:
        					print('Lost track of object with ID = {}'.format(procid[conn]))
        				if misscount[conn] >= MISSLIMIT:
        					remove_list.append(i)
        					print('Reached miss limit')
        					killprocesses(runningprocesses, cols, remove_list)

        					print('Opened new process')
        					p = subprocess.Popen('python a.py', shell = True, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
        					freeprocesses.append(p)
        					cols.append([0, 0, 255])

        					new_bb = performDetect(1,paths[global_frame-1])
        					for r in range(len(new_bb)):
        						print(new_bb[r])
        						print('\n')
        					prevcoords.pop(i)
        					print(prevcoords)

        					for k in range(len(new_bb)):
        						newcoords = new_bb[k][2]
        						counter = 0
        						for l in range(len(prevcoords)):
        							if intersection(prevcoords[l],transformBB(new_bb[k][2])) > 0.85:
        								continue
        							else:
        								counter = counter + 1
        						if counter == len(prevcoords):
        							break
        					print(newcoords)
        					newcoords = transformBB(newcoords)
        					addbox(newcoords, global_frame-1, freeprocesses, runningprocesses)
        					remove_list = []

        					listprocesses = list(enumerate(runningprocesses))
        					nextconn = listprocesses[i][1]
        					newmsg = nextconn.stdout.readline()
        					newmsg = [ float(z) for z in newmsg.split(' ') ]
        					confidence_score.append(float(newmsg[-1]))
        					newmsg = newmsg[:4]
        					newmsg = [int(m) for m in newmsg]
        					coords.append(newmsg)
        					'''
        					confidence_score.append(float(msg[-1]))
        					msg = msg[:4]
        					msg = [ int(m) for m in msg]
        					coords.append(msg)
        					killprocesses(runningprocesses, cols, remove_list)
        					new_p = subprocess.Popen('python a.py', shell = True, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
        					freeprocesses.append(new_p)
        					new_bbox = performDetect(0,paths[0])
        					firstframe.append(gkobal_frame)
        					addbox(new_bbox[i], global_frame, freeprocesses, runningprocesses)
        					'''
        					continue
        			else:
        				if misscount[conn] > 5:
        					print('Object with ID = {} found'.format(procid[conn]))
        				misscount[conn] = 0
        			confidence_score.append(float(msg[-1]))
        			msg = msg[:4]
        			msg = [ int(m) for m in msg]
        			coords.append(msg)
        			if(i==0):
        				yolo_bb.append(performDetect(0,paths[count]))
        				count = count + 1
        		prevcoords = coords
        		killprocesses(runningprocesses, cols, remove_list)
        		if not runningprocesses:
        			flag = 0
        			continue
        		meem_path = drawrectangles(paths[global_frame-1], coords, cols, confidence_score, global_frame, runningprocesses)
        		yolo_img = Image.open(yolo_bb[count-1])
        		meem_img = Image.open(meem_path)
        		yolo_img = yolo_img.resize((576, 432), Image.ANTIALIAS)
        		meem_img = meem_img.resize((576, 432), Image.ANTIALIAS)
        		yolo_img = ImageTk.PhotoImage(yolo_img)
        		meem_img = ImageTk.PhotoImage(meem_img)
        		panel.configure(image = yolo_img)
        		panel1.configure(image = meem_img)
        		root.update_idletasks()
        		time.sleep(0.01)
        		#count += 1
        		global_frame += 1

        except KeyboardInterrupt:
        	killprocesses(runningprocesses, cols, [i for i in range(len(runningprocesses))])
	panel.configure(image = detectionComplete)
	panel1.configure(image = trackingComplete)
	root.update_idletasks()
	time.sleep(0.03)

detectionUsingYolo = ImageTk.PhotoImage(Image.open('C:/darknet/build/darknet/x64/Detection Using Yolo.png'))
trackingUsingMEEM = ImageTk.PhotoImage(Image.open('C:/darknet/build/darknet/x64/Tracking Using MEEM.png'))
detectionComplete = ImageTk.PhotoImage(Image.open('C:/darknet/build/darknet/x64/Detection Complete.png'))
trackingComplete = ImageTk.PhotoImage(Image.open('C:/darknet/build/darknet/x64/Tracking Complete.png'))
    
#Frame Initialization
titleFrame = tk.Frame(root,width=1280,height=100)
btnFrame1 = tk.Frame(root,width=1280,height=100)
btnFrame2 = tk.Frame(root,width=1280,height=100)
panelFrame = tk.Frame(root,width=1280,height=480,padx=15, pady=15)

#Root Layout
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

#Frame Layout
titleFrame.grid(row=0,columnspan=1)
btnFrame1.grid(row=1,sticky="ew")
btnFrame2.grid(row=2,sticky="ew")
panelFrame.grid(row=3,columnspan=2)

#Component Initialization
title = tk.Label(titleFrame, text="Multi Object Detection and Tracking Tool", font=("Arial Bold",45))
path = tk.Button(btnFrame1, text ="Select Diretory", width=30, height=5, command= lambda: getPath())
num = tk.Button(btnFrame1, text ="Select Number of Objects", width=30, height=5, command= lambda: getNum())
start = tk.Button(btnFrame2, text ="Start", width=30, height=5, command= lambda: startTrack())
conv = tk.Button(btnFrame2, text ="Save Output Video", width=30, height=5, compound=tk.RIGHT, command= lambda: writeVideo())
panel = tk.Label(panelFrame, image = detectionUsingYolo, compound=tk.LEFT)
panel1 = tk.Label(panelFrame, image = trackingUsingMEEM, compound=tk.RIGHT)

#Component Layout
title.grid(row=0)
path.grid(column=0,row=1,padx=(220,45))
num.grid(column=1,row=1,padx=(355,0))
start.grid(column=0,row=2,padx=(220,45))
conv.grid(column=1,row=2,padx=(355,0))
panel.grid(row=3,column=0)
panel1.grid(row=3,column=1)

root.mainloop()
