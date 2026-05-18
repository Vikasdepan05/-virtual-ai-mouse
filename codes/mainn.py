import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[8]
                middle_finger_tip = hand_landmarks.landmark[12]

                ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

                # Move mouse
                screen_x = int(index_finger_tip.x * screen_w)
                screen_y = int(index_finger_tip.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

                # Clicking condition
                if abs(ix - mx) < 30 and abs(iy - my) < 30:
                    pyautogui.click()
                    pyautogui.sleep(0.2)

        cv2.imshow("AI Virtual Mouse", frame)

        if cv2.waitKey(1) == 27:  # Esc to exit
            break

cap.release()
cv2.destroyAllWindows()




