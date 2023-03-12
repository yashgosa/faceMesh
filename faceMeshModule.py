# importing the required modules

import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode,
        self.max_num_faces = max_num_faces,
        self.refine_landmarks = refine_landmarks,
        self.min_detection_confidence = min_detection_confidence,
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1) # Creating a mediapipe face mesh object
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1) # setting the color thickness circle_radius of face mesh

    def createMesh(self, img, draw = True):
        faces=[]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks: #if any face detected
            for faceLms in self.results.multi_face_landmarks: #Looping through each face
                # drawing the landmarks on the frame
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face= []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.5, (0, 255, 0), 1)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.createMesh(img)
        cTime = time.time()
        if faces:
            print(faces)
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()