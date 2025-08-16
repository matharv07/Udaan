import cv2
import mediapipe as mp

def raisedCount(map):
    tipIds = (8, 12, 16, 20)
    countDic = {'uCount' : 0, 'dCount' : 0, 'lCount' : 0, 'rCount' : 0, 'orient' : 'none'}
    count = max(countDic['uCount'], countDic['dCount'], countDic['lCount'], countDic['rCount'])
    for tip in tipIds:
        yfingerPos = [map.landmark[tip].y, map.landmark[tip-1].y, map.landmark[tip-2].y, map.landmark[tip-3].y]
        xfingerPos = [map.landmark[tip].x, map.landmark[tip-1].x, map.landmark[tip-2].x, map.landmark[tip-3].x]
        uCheck = True
        for i in (0, 2):
            if yfingerPos[i] > yfingerPos[i+1] or count == 4:
                uCheck = False
                break
        if uCheck:
            countDic['uCount'] += 1
            if countDic['uCount'] > count:
                countDic['orient'] = 'uCount'
                count = countDic['uCount']
        """
        dCheck = True
        for i in (0, 2):
            if yfingerPos[i] < yfingerPos[i + 1] or count == 4:
                dCheck = False
                break
        if dCheck:
            countDic['dCount'] += 1
            if countDic['dCount'] > count:
                countDic['orient'] = 'dCount'
                count = countDic['dCount']
        lCheck = True
        for i in (0, 2):
            if xfingerPos[i] > xfingerPos[i + 1] or count == 4:
                lCheck = False
                break
        if lCheck:
            countDic['lCount'] += 1
            if countDic['lCount'] > count:
                countDic['orient'] = 'lCount'
                count = countDic['lCount']
        rCheck = True
        for i in (0, 2):
            if xfingerPos[i] < xfingerPos[i + 1] or count == 4:
                rCheck = False
                break
        if rCheck:
            countDic['rCount'] += 1
            if countDic['rCount'] > count:
                countDic['orient'] = 'rCount'
                count = countDic['rCount']
        """
    print(count, "fingers raised.", countDic['orient'], "orientation.")
    return count

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
        fingers = 0
        if detect.multi_hand_landmarks:
            for hand_lms in detect.multi_hand_landmarks:
                draw.draw_landmarks(image, hand_lms, handMP.HAND_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))
                fingers = raisedCount(hand_lms)
        cv2.putText(image, f"Fingers: {fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', image)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()