import os
import cv2

import os
import cv2

def read_video(video_path):
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        yield frame

    video.release()


def save_Video(frames, output_video_path: str, fps: float = 24.0):
    if not frames:
        return
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
