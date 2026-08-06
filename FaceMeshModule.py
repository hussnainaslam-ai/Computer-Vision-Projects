import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = float(minDetectionCon)
        self.minTrackCon = float(minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    # FIX: FACE_CONNECTIONS replaced with FACEMESH_TESSELATION
                    self.mpDraw.draw_landmarks(
                        img, 
                        faceLms, 
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.drawSpec, 
                        self.drawSpec
                    )
                face = []
                ih, iw, _ = img.shape
                for id, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture("Source/4.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)

    while True:
        success, img = cap.read()
        if not success:
            print("Video stream finished or failed to read frame.")
            break

        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(f"Detected Faces: {len(faces)} | Landmarks in face 0: {len(faces[0])}")

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Face Mesh Detector", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()