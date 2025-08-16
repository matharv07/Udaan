import cv2
import numpy as np
import mediapipe as mp
import os

draw = mp.solutions.drawing_utils
handMP = mp.solutions.hands
cap = cv2.VideoCapture(0)
subdir = 'out'
n_frames_save = 64
iteration_counter = n_frames_save + 1
folder_counter = 1
check = True
itr = 0
X, y = [], []
mapping = {'fist': 0, 'stawp': 1, 'yoo': 2, 'out': 3 }
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
with handMP.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, static_image_mode=True) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        #frame = cv2.flip(frame, 1)
        sample = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if detect.multi_hand_landmarks:
            for hand_lms in detect.multi_hand_landmarks:
                draw.draw_landmarks(image, hand_lms, handMP.HAND_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))
        cv2.imshow('frame', frame)
        cv2.imshow('image', image)
        print(iteration_counter, "vs", n_frames_save, ':', itr)
        seq_folder_path = os.path.join('data', subdir, f'sequence{folder_counter}')
        if check and itr > 100:
            print('entered')
            iteration_counter = 1
            os.makedirs(seq_folder_path)
            check = False
        if iteration_counter < n_frames_save + 1:
            print('writing', iteration_counter)
            i = iteration_counter
            path = os.path.join(seq_folder_path, f'{subdir}_sequence{folder_counter}_frame{i}.jpg')
            cv2.imwrite(path, image)
            if detect.multi_hand_landmarks:
                for hand_lms in detect.multi_hand_landmarks:
                    for lm in hand_lms.landmark:
                        sample.extend([lm.x, lm.y])
                    X.append(sample)
                    y.append(mapping[subdir])
            print(os.path.join(seq_folder_path, f'{subdir}_sequence{folder_counter}_frame{i}.jpg'))
            if iteration_counter == n_frames_save:
                print(f'Images for sequence {folder_counter - 1} saved.')
            iteration_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        itr += 1
cap.release()
cv2.destroyAllWindows()
X = np.array(X)
y = np.array(y)
print(X)
print(y)
np.savez(os.path.join('data', f'data_{subdir}.npz'), X=X, y=y)
