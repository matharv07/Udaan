import cv2
import mediapipe as mp

def checkOrientation(map):
    tipIds = (8, 12, 16, 20) #8 - index, 12, middle, 16 - ring, 20, pinky
    isFingerRaised = {'8' : False, '12' : False, '16' : False, '20' : False}
    for tip in tipIds:
        yfingerPos = [map.landmark[tip].y, map.landmark[tip - 1].y, map.landmark[tip - 2].y, map.landmark[tip - 3].y]
        print(yfingerPos)
        check = True
        for i in (0, 2):
            if yfingerPos[i] > yfingerPos[i + 1]:
                check = False
                break
        if check:
            isFingerRaised[str(tip)] = True
            print(tip, "raised.")
        else:
            print(tip, "down.")
    return isFingerRaised

draw = mp.solutions.drawing_utils
handMP = mp.solutions.hands
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
with handMP.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fingers = {'8' : False, '12' : False, '16' : False, '20' : False}
        if detect.multi_hand_landmarks:
            for hand_lms in detect.multi_hand_landmarks:
                draw.draw_landmarks(image, hand_lms, handMP.HAND_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))
                fingers = checkOrientation(hand_lms)
        cv2.putText(image, str(fingers['8']) + str(fingers['12']) + str(fingers['16']) + str(fingers['20']), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', image)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()