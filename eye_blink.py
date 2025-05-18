# import cv2
# import dlib
# import imutils
# from scipy.spatial import distance as dist
# from imutils import face_utils
# import pygame
# import os



# blink_threshold = 0.5
# frame_success = 2
# frame_count = 0
# blink_count = 0  


# pygame.mixer.init()
# alert_sound = 'alert.wav'
# print("Sound file exists:", os.path.exists('alert.wav'))

# cam = cv2.VideoCapture(0)

# (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
# (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# face_detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

# def EAR_calculate(eye):
#     a1 = dist.euclidean(eye[1], eye[5])
#     a2 = dist.euclidean(eye[2], eye[4])
#     m = dist.euclidean(eye[0], eye[3])
#     EAR = (a1 + a2) / m
#     return EAR

# def eyeLandmark(img, eyes):
#     for eye in eyes:
#         x1, x2 = (eye[1], eye[5])
#         x3, x4 = (eye[0], eye[3])
#         cv2.line(img, x1, x2, (178, 200, 226), 2)
#         cv2.line(img, x3, x4, (178, 200, 226), 2)
#         for i in range(6):
#             cv2.circle(img, tuple(eye[i]), 3, (200, 223, 0), -1)
#     return img

# while True:

#     if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
#         cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     ret, frame = cam.read()
#     frame = imutils.resize(frame, width=512)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = face_detector(rgb)

#     for face in faces:
#         shape = predictor(rgb, face)
#         shape = face_utils.shape_to_np(shape)
        
#         for lm in shape:
#             cv2.circle(frame, (lm), 3, (10, 2, 200))

#         lefteye = shape[L_start:L_end]
#         righteye = shape[R_start:R_end]

#         left_EAR = EAR_calculate(lefteye)
#         right_EAR = EAR_calculate(righteye)

#         img = frame.copy()
#         img = eyeLandmark(img, [lefteye, righteye])

#         avg = (left_EAR + right_EAR) / 2

#         if avg < blink_threshold:
#             frame_count += 1
#             if frame_count >= 50:  # customize based on your FPS
#                 if not pygame.mixer.get_busy():
#                     pygame.mixer.Sound(alert_sound).play()
#         else:
#             if frame_count >= frame_success:
#                 blink_count += 1  # 🔸 Count the blink
#                 cv2.putText(img, 'Blink Detected', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (233, 0, 189), 1)
#             frame_count = 0

#         # 🔸 Show blink count on top right
#         cv2.putText(img, f'Blinks: {blink_count}', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow('Eye Blinking', img)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()





import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame
import os
import time  # ⬅️ Only this added import

blink_threshold = 0.5
frame_success = 2
frame_count = 0
blink_count = 0  

pygame.mixer.init()
alert_sound = 'alert.wav'
print("Sound file exists:", os.path.exists('alert.wav'))

cam = cv2.VideoCapture(0)

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat')

last_blink_time = time.time()  # ⬅️ Added to track last blink

def EAR_calculate(eye):
    a1 = dist.euclidean(eye[1], eye[5])
    a2 = dist.euclidean(eye[2], eye[4])
    m = dist.euclidean(eye[0], eye[3])
    EAR = (a1 + a2) / m
    return EAR

def eyeLandmark(img, eyes):
    for eye in eyes:
        x1, x2 = (eye[1], eye[5])
        x3, x4 = (eye[0], eye[3])
        cv2.line(img, x1, x2, (178, 200, 226), 2)
        cv2.line(img, x3, x4, (178, 200, 226), 2)
        for i in range(6):
            cv2.circle(img, tuple(eye[i]), 3, (200, 223, 0), -1)
    return img

while True:

    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cam.read()
    frame = imutils.resize(frame, width=512)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb)

    for face in faces:
        shape = predictor(rgb, face)
        shape = face_utils.shape_to_np(shape)
        
        for lm in shape:
            cv2.circle(frame, (lm), 3, (10, 2, 200))

        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        left_EAR = EAR_calculate(lefteye)
        right_EAR = EAR_calculate(righteye)

        img = frame.copy()
        img = eyeLandmark(img, [lefteye, righteye])

        avg = (left_EAR + right_EAR) / 2

        if avg < blink_threshold:
            frame_count += 1
            # ⬇️ Check time since last blink
            if time.time() - last_blink_time >= 5:
                if not pygame.mixer.get_busy():
                    pygame.mixer.Sound(alert_sound).play()
        else:
            if frame_count >= frame_success:
                blink_count += 1
                last_blink_time = time.time()  # ⬅️ Reset timer on blink
                cv2.putText(img, 'Blink Detected', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (233, 0, 189), 1)
            frame_count = 0

        # Show blink count
        cv2.putText(img, f'Blinks: {blink_count}', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Eye Blinking', img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
