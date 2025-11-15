import cv2
import mediapipe as mp
import time

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose(
  static_image_mode=False,
  smooth_landmarks=True,
  # model_complexity=0,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
)

cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)
cTime=0
pTime=0

while True:
  success,img=cap.read()

  imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  results=pose.process(imgRGB)

  # print(results.pose_landmarks)
  if results.pose_landmarks:
    mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    for id,lm in enumerate(results.pose_landmarks.landmark):
      h,w,c =img.shape
      print(id,lm)
      cx,cy= int(lm.x*w) ,int(lm.y*h)
      cv2.circle(img,(cx,cy),5,(64,164,89),cv2.FILLED)

  cTime = time.time()
  fps=1/(cTime-pTime) 
  pTime=cTime

  cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),3)
  cv2.imshow("Image",img)
  cv2.waitKey(1)
