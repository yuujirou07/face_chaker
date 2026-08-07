from pathlib import Path
import time

import cv2
import mediapipe as mp
import evdev
from evdev import UInput, ecodes as e
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from enum import Enum
class hand_side(Enum):
    left = 1
    right = 2



#モデル読み込み
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def draw_hand_landmarks(image, result):
    for hand_landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )
    
def output_hand_state(hand_state):
    if hand_state.handedness:
        print(hand_state.handedness[0][0].index)


def get_handednes_num(hand_state,num):
    hand_num = len(hand_state.handedness)
    if hand_num < 0:return -2
    for i in range(hand_num):
        if hand_state.handedness[i][0].index == num:
            return i
    return -1




def mcheck_hands_point(result,image,hands_side):
    side = 0
    if hands_side == hand_side.left:
        side = 0
    elif hands_side == hand_side.right:
        side = 1
        
    num = get_handednes_num(result,side)
    if num >= 0:
        hand_land_mark = result.hand_landmarks[num] 
        h,w,c= image.shape

        thumb = [0]*2#親指
        index_finger = [0]*2#人差し指
        is_see = 0
        for i in range(len(hand_land_mark)):

            x = int(hand_land_mark[i].x * w)
            y = int(hand_land_mark[i].y * h)

            if i == 4:
                thumb[0] = x
                thumb[1] = y
                is_see += 1
            elif i == 8:
                index_finger[0] = x
                index_finger[1] = y
                is_see+= 1

            if is_see >= 2:
                f =  cv2.line(image, (thumb), (index_finger), (0, 0, 255), 2)
                pos1 = np.array(thumb)
                pos2 = np.array(index_finger)
                diff = pos1 - pos2
                if diff[0] < 30 and diff[1] < 30:
                    return 1

            

            r,g,b = (0,0,0)
            if i == 8:
                r  = 255
            else: b = 255
                
            cv2.circle(image,(x,y),3,(r,g,b),-1)
    return 0    
            

    


def main():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    camera = cv2.VideoCapture(0)
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("カメラから画像を取得できませんでした。")
                    continue

                frame = cv2.flip(frame, 1)

                # OpenCVはBGR、MediaPipeはRGB形式で画像を扱う。
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # VIDEOモードでは、各フレームに単調増加する時刻が必要。
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                
                if len(result.handedness) > 0:
                    result = check_hands_point(result,frame,hand_side.left)
                    #if result == 1:
                        



               
                cv2.imshow("Left Hand Landmarks", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
