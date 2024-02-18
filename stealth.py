import cv2
from ultralytics import YOLO
import math

classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
              'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

model = YOLO('yolov8l.pt')
video_path = "WHATSAAP_ASSIGNMENT.mp4"

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output/outputvid_ball.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (255, 255, 255)
text_offset = 30

prev_ball_center = None
dribble_threshold = 55
dribble_count = 0
frame_skip = 3
frame_count = 0

velocity_list = []
frequency_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(frame, conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = int(box.cls[0])
            cc = classNames[c]
            x1, y1, x2, y2 = box.xyxy[0]

            if cc == "sports ball":
                ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                if prev_ball_center is not None:
                    if ball_center[1] > prev_ball_center[1]:
                        if ball_center[1] - prev_ball_center[1] >= dribble_threshold:
                            dribble_count += 1

                prev_ball_center = ball_center

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                if len(velocity_list) > 0:
                    distance = ball_center[1] - velocity_list[-1][1]
                    time_elapsed = (frame_count - 1) * frame_skip / cap.get(cv2.CAP_PROP_FPS)
                    velocity = distance / time_elapsed
                    velocity_list.append((frame_count, ball_center[1], velocity))

                    frequency = 1 / time_elapsed
                    frequency_list.append((frame_count, ball_center[1], frequency))
                else:
                    velocity_list.append((frame_count, ball_center[1], 0))
                    frequency_list.append((frame_count, ball_center[1], 0))

    cv2.putText(frame, f"Dribbles: {dribble_count}", (50, 650), font, font_scale, text_color, font_thickness)
    if len(velocity_list) > 0:
        cv2.putText(frame, f"Velocity: {velocity_list[-1][2]:.2f} m/s", (50, 700), font, font_scale, text_color, font_thickness)
    if len(frequency_list) > 0:
        cv2.putText(frame, f"Frequency: {frequency_list[-1][2]:.2f} Hz", (50, 750), font, font_scale, text_color, font_thickness)

    output_video.write(frame)
    cv2.imshow("Video with Bounding Boxes", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
