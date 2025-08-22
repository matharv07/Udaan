from time import sleep
import cv2
import mediapipe as mp

prevFrame = [0, 0]
def checkOrientation(map, prevFrame):
    tipId = 0
    detectBarrier = 10                           #cameraspecific
    currFrame = (map.landmark[tipId].x * 480, map.landmark[tipId].y * 640)
    #sleep(0.005)
    horzId, vertId = 0, 0
    if currFrame[1] - prevFrame[1] > detectBarrier:
        vertId = 1
    elif currFrame[1] - prevFrame[1] < -detectBarrier:
        vertId = 2
    if currFrame[0] - prevFrame[0] > detectBarrier:
        horzId = 3
    elif currFrame[0] - prevFrame[0] < -detectBarrier:
        horzId = 4
    if abs(currFrame[0] - prevFrame[0]) > abs(currFrame[1] - prevFrame[1]):
            return horzId, currFrame
    else:
            return vertId, currFrame
def checkOpen(map):
    tipIds = (8, 12, 16, 20)  # 8 - index, 12, middle, 16 - ring, 20, pinky
    openOrNot = 1
    for tip in tipIds:
        yfingerPos = [map.landmark[tip].y, map.landmark[tip - 1].y, map.landmark[tip - 2].y, map.landmark[tip - 3].y, 0]
        for i in (0, 2):
            if yfingerPos[i] > yfingerPos[i + 1]:
                openOrNot = 2
                break
    return openOrNot

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
        moveId = 0
        movesList = ['rest', 'down', 'up', 'right', 'left']
        openOrNot = 0
        fist = ['invisible', 'open', 'closed']
        if detect.multi_hand_landmarks:
            for hand_lms in detect.multi_hand_landmarks:
                draw.draw_landmarks(image, hand_lms, handMP.HAND_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))
                moveId, prevFrame = checkOrientation(hand_lms, prevFrame)
                openOrNot = checkOpen(hand_lms)
                print(prevFrame)
        cv2.putText(image, 'Hand ' + movesList[moveId] + ' is ' + fist[openOrNot], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print('Hand ' + movesList[moveId])
        cv2.imshow('frame', image)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
