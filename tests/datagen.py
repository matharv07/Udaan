import cv2
import mediapipe as mp

def count_fingers(hand_landmarks):
    """Count raised fingers using MediaPipe landmarks."""
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    fingers = 0

    # Check fingers (except thumb)
    for tip_id in tips:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers += 1

    # Thumb (x position check)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers += 1
    print("n/fingers", fingers)
    return fingers

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(2)

command_map = {
    1: "F",  # Forward
    2: "B",  # Backward
    3: "L",  # Left
    4: "R",  # Right
    5: "S"   # Stop
}

last_command = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

    # Display finger count
    cv2.putText(frame, f"Fingers: {finger_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    # Send command to ESP32 over Wi-Fi
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Clean up
cap.release()
cv2.destroyAllWindows()