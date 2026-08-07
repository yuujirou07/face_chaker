from pathlib import Path
import json
import time

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
MATCH_THRESHOLD = 0.05


right_outer_eye_region_idx = 446
left_outer_eye_region_idx = 226

middle_top_rip_idx = 0
middle_nose_top_idx = 1

middle_between_eyes_idx = 168

draw_landmark_idx_list = [right_outer_eye_region_idx, left_outer_eye_region_idx,
                          middle_top_rip_idx, middle_nose_top_idx,
                          middle_between_eyes_idx]

def set_face_map(image, result):
    if not result.face_landmarks:
        return []

    i = 0
    middle_rip_pos = [0]*2
    nose_top = [0]*2
    face_landmarks = result.face_landmarks[0]
  
    h,w,c = image.shape
    land_mark_num = len(result.face_landmarks[0])
    point_list = [[0],[0]]* land_mark_num
    for k in range(land_mark_num):
        x = int(w * face_landmarks[k].x)
        y = int(h * face_landmarks[k].y)
        if k in draw_landmark_idx_list:
            r,g,b = 0,0,255
            cv2.circle(image,(x,y),1,(r,g,b),-1)

        point_list[k] = [face_landmarks[k].x,face_landmarks[k].y]

    return point_list


def get_point_distances(point_list):
    eye_width = math.dist(
        point_list[right_outer_eye_region_idx],
        point_list[left_outer_eye_region_idx],
    )
    if eye_width == 0:
        return {}

    distances = {}
    for i, first_idx in enumerate(draw_landmark_idx_list):
        first_point = point_list[first_idx]
        for second_idx in draw_landmark_idx_list[i + 1:]:
            second_point = point_list[second_idx]
            distances[f"{first_idx}-{second_idx}"] = (
                math.dist(first_point, second_point) / eye_width
            )
    return distances


def save_face_data(name, distance_samples):
    average_distances = {
        key: sum(sample[key] for sample in distance_samples) / len(distance_samples)
        for key in distance_samples[0]
    }
    file_path = Path(__file__).with_name(f"{name}_face_landmarks.json")
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "landmark_indices": draw_landmark_idx_list,
                "reference_landmarks": [
                    right_outer_eye_region_idx,
                    left_outer_eye_region_idx,
                ],
                "sample_count": len(distance_samples),
                "average_distance_ratios": average_distances,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return file_path


def get_saved_distance_ratios(data):
    distance_ratios = data.get("average_distance_ratios")
    if distance_ratios:
        return distance_ratios

    distances = data.get("average_distances")
    eye_width = distances.get(f"{right_outer_eye_region_idx}-{left_outer_eye_region_idx}")
    if eye_width is None or eye_width == 0:
        return {}
    return {key: value / eye_width for key, value in distances.items()}


def identify_face(distances):
    best_name = None
    best_error = float("inf")
    for file_path in Path(__file__).parent.glob("*_face_landmarks.json"):
        try:
            with file_path.open(encoding="utf-8") as file:
                saved_distances = get_saved_distance_ratios(json.load(file))
        except (OSError, json.JSONDecodeError):
            continue

        if saved_distances.keys() != distances.keys():
            continue

        error = sum(
            abs(distances[key] - saved_distances[key]) for key in distances
        ) / len(distances)
        if error < best_error:
            best_error = error
            best_name = file_path.stem.removesuffix("_face_landmarks")

    if best_error <= MATCH_THRESHOLD:
        return best_name, best_error
    return None, best_error


def check_face_neutral(face_blendshapes):
    if not face_blendshapes:
        return 0

    expression_scores = [
        category.score for category in face_blendshapes[0]
        if category.category_name != "_neutral"
    ]
    if max(expression_scores, default=1.0) > 0.95:
        return 1
    return 0

    


            


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,output_facial_transformation_matrixes=1,
        num_faces=1,output_face_blendshapes=True,
    )
    camera = cv2.VideoCapture(0)
    ipt = 0
    name = []
    while(1):
        print("顔を追加する場合は１")
        print("顔認証テストの場合は２")
        print("を押してください")
        ipt = int(input())
        if ipt != 1 and ipt != 2:
            print("1か2を押してください")
            continue
        break
        
    if ipt == 1:
        while(1):
            print("追加する顔の名前を入力してください")
            name = str(input())
            print(f"[{name}]でいいですか？[Y] [N]")
            is_ok = str(input()) 
            if is_ok == "n" or is_ok == "N":
                continue
            break
    elif ipt == 2:
        print("判定中です")

    distance_samples = []
    last_recognition_result = ""

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("カメラから画像を取得できませんでした。")
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
                point_list = set_face_map(frame, result)

                if point_list:
                    face_neutral = check_face_neutral(result.face_blendshapes)
                    if face_neutral <= 0:
                        print("顔を動かさないでください")
                    elif ipt == 1:
                        distances = get_point_distances(point_list)
                        if distances:
                            distance_samples.append(distances)
                            print(f"保存中: {len(distance_samples)}/100")
                            if len(distance_samples) == 100:
                                file_path = save_face_data(name, distance_samples)
                                print(f"平均距離を保存しました: {file_path}")
                                break
                    elif ipt == 2:
                        distances = get_point_distances(point_list)
                        if distances:
                            person_name, error = identify_face(distances)
                            if person_name is None:
                                recognition_result = "Unknown face"
                                color = (0, 0, 255)
                            else:
                                recognition_result = f"Recognized: {person_name}"
                                color = (0, 255, 0)

                            cv2.putText(
                                frame,
                                recognition_result,
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                2,
                            )
                            if recognition_result != last_recognition_result:
                                print(f"{recognition_result} (error={error:.4f})")
                                last_recognition_result = recognition_result

                cv2.imshow("Face Landmarks", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
