# -*- coding: utf-8 -*-


#euclidean 거리 측정을 위한 라이브러리
from scipy.spatial import distance as dist
#cropping을 위한 라이브러리
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
#알람설정 라이브러리
import playsound
import imutils
import time
#machine learing 라이브러리
import dlib
#opencv 사용
import cv2

def sound_alarm(path):
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 
EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

#dlib의 정면 얼굴 검출기 사용
detector = dlib.get_frontal_face_detector()
#data파일
predictor = dlib.shape_predictor('face_landmarks.dat')

#양쪽 눈 좌표찾기
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    #비디오에서 얼굴부분 cropping
	frame = vs.read()
	frame = imutils.resize(frame, width=450)

    #흑백처리
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
        #ear계산
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
        #양쪽 눈 육각형으로 표시
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
					ALARM_ON = True
					
				cv2.putText(frame, "detect!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			COUNTER = 0
			ALARM_ON = False
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
