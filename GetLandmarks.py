import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

#Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
#         Recolour image to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#         Make detections
        results = holistic.process(image)
    
#     Recolour image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     Draw landmarks

#       1.  face
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0,0,255),thickness=1, circle_radius=1),mp_drawing.DrawingSpec(color=(0,0,255),thickness=1, circle_radius=1))
#       2.  pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#       3.  left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#       4.  right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    
#         print(results.face_landmarks)
        cv2.imshow('Raw Video Feed', image)
#     
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()