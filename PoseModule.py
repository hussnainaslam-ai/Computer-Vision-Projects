import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Pose model for multi-person detection
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # MediaPipe Pose and Drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def findPoses(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPositions(self, img, draw=True):
        lmLists = []
        if self.results.pose_landmarks:
            height, width, _ = img.shape
            for lm in self.results.pose_landmarks.landmark:
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmLists.append((cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        return lmLists


# Main function to capture video and perform multi-person detection
def main():
    cap = cv2.VideoCapture("Source/12.mp4")  # Use a video with multiple people
    pTime = 0
    detector = poseDetector()  # Initialize the pose detector

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect poses and draw landmarks
        img = detector.findPoses(img, draw=True)
        lmLists = detector.findPositions(img, draw=True)

        if lmLists:
            print(f"Number of landmarks detected: {len(lmLists)}")

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Resize the frame for better visualization
        img = cv2.resize(img, (1280, 720))

        # Display the image
        cv2.imshow("Multi-Person Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on pressing 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
