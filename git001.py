
''' Blue and Red tracking + kinda writing program

added a direction display for Blue.

Coded by SuFei(粛飛)			mail : japanpoosa@gmail.com
'''
#################################################################################################################################################################################


from collections import deque
import numpy as np 
import argparse as ag 
import imutils
import cv2

ap=ag.ArgumentParser()
ap.add_argument("-v","--video",help="Video file path")
ap.add_argument("-b","--buffer",type=int ,default=32,help="maximum buffer size")
args=vars(ap.parse_args())

Low_Blue=(90,100,100)
Upper_Blue=(130,255,255)
Low_Red= (150,150,0)
Upper_Red= (255,255,255)

pts_R=deque(maxlen=args["buffer"])
counter_R=0
(dx_R,dy_R)=(0,0)
direction_R=""

pts_b=deque(maxlen=args["buffer"])
counter_b=0
direction_b=""
(dx_b,dy_b)=(0,0)

if not args.get("Video",False):
	cam=cv2.VideoCapture(0)

else:
	cam=cv2.VideoCapture(args["Video"])	

while True:
	(ret,frame)=cam.read()

	if args.get("video") and not ret:
		break

	frame=imutils.resize(frame,width=600)
	blur=cv2.GaussianBlur(frame,(11,11),0)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	mask_b=cv2.inRange(hsv,Low_Blue,Upper_Blue)#blue
	cv2.imshow("maskb",mask_b)
	mask_R=cv2.inRange(hsv,Low_Red,Upper_Red)
	cv2.imshow("maskr",mask_R)
	mask_R=cv2.erode(mask_R,None,iterations=2)#red
	mask_b=cv2.erode(mask_b,None,iterations=2)#blue
	mask_R=cv2.dilate(mask_R,None,iterations=2)#red
	mask_b=cv2.dilate(mask_b,None,iterations=2)#blue
	cnts_R=cv2.findContours(mask_R.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	cnts_b=cv2.findContours(mask_b.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center_R=None
	center_b=None

	if len(cnts_R)>0:
		c_R=max(cnts_R,key=cv2.contourArea)
		((x_R,y_R),radius_R)=cv2.minEnclosingCircle(c_R)
		M_R=cv2.moments(c_R)
		center_R=(int(M_R["m10"]/M_R["m00"]),int(M_R["m01"]/M_R["m00"]))

		if radius_R>10:
			cv2.circle(frame,(int(x_R),int(y_R)),int(radius_R),	(0,0,255),2)
			cv2.circle(frame,center_R,5,	(0,0,255),-1)
			pts_R.appendleft(center_R)

	for i in np.arange(1,len(pts_R)):
		if pts_R[i-1] is None or pts_R[i] is None:
			continue

		if(counter_R>=50 and i==1 and pts_R[-10] is not None):
			dx_R=pts_R[-10][0]-pts_R[i][0]
			dy_R=pts_R[-10][1]-pts_R[i][1]
			(dirX_R,dirY_R)=("","")

			if np.abs(dy_R)>20:
				dirX_R="East" if np.sign(dx_R)==1 else "West"

			if np.abs(dy_R)>20:
				dirY_R="North" if np.sign(dy_R)==1 else "South"

			if dirX_R!="" and dirY_R!="":
				direction_R="{}-{}".format(dirY_R,dirX_R)

			else:
				direction_R=dirX_R if dirX_R!="" else dirY_R


		thickness_R=int(np.sqrt(args["buffer"]/float(i+1))*5.5)
		cv2.line(frame,pts_R[i-1],pts_R[i],	(0,0,255),thickness_R)	

	cv2.putText(frame,direction_R,(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),3)
	cv2.putText(frame,"dx:{},dy:{}".format(dx_R,dy_R),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)

	# #---------------------------------------------------------------------------------------------------------------------------------
	if len(cnts_b)>0:

		c_b=max(cnts_b,key=cv2.contourArea)
		((x,y),radius_b)=cv2.minEnclosingCircle(c_b)
		M_b=cv2.moments(c_b)
		center_b=(int(M_b["m10"]/M_b["m00"]),int(M_b["m01"]/M_b["m00"]))

		if radius_b>1:

			cv2.circle(frame,(int(x),int(y)),int(radius_b),(255,0,0),2)
			cv2.circle(frame,center_b,5,(255,0,0),-1)
			pts_b.appendleft(center_b)

	for i in np.arange(1,len(pts_b)):

		if pts_b[i-1] is None or pts_b[i] is None:
			continue

		if(counter_b>=50 and i==1 and pts_b[-10] is not None):

			dx_b=pts_b[-10][0]-pts_b[i][0]
			dy_b=pts_b[-10][1]-pts_b[i][1]
			(dirX_b,dirY_b)=("","")

			if np.abs(dy_b)>20:
				dirX_b=("East") if np.sign(dx_b)==1 else ("West")


			if np.abs(dy_b)>20:
				dirY_b=("North") if np.sign(dy_b)==1 else ("South")

			if dirX_b!="" and dirY_b!="":
				direction_b="{}-{}".format(dirY_b,dirX_b)

			else:
				direction_b=dirX_b if dirX_b!="" else dirY_b


		thickness_b=int(np.sqrt(args["buffer"]/float(i+1))*5.5)
		cv2.line(frame,pts_b[i-1],pts_b[i],(255,0,0),thickness_b)	


	cv2.putText(frame,direction_b,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),3)
	cv2.putText(frame,"dx:{},dy:{}".format(dx_b,dy_b),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,0,0),1)

	mask=(mask_R+mask_b)
	res=cv2.bitwise_and(frame,frame,mask)
	cv2.imshow("frame",res)
	key=cv2.waitKey(30) &0xff
	counter_b+=1

	if key==ord("q"):
		break

cam.release()
cv2.destroyAllWindows()		




