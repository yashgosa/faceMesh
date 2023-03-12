# importing the required modules

import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True) # Creating a mediapipe face mesh object
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1) # setting the color thickness circle_radius of face mesh

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks: #if any face detected
        for faceLms in results.multi_face_landmarks: #Looping through each face
            # drawing the landmarks on the frame
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)