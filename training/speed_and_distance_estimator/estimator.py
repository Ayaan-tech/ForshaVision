import sys
import cv2
import numpy as np
sys.path.append('../')
from training.utils import measure_distance, measure_xy_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object_name, object_tracks in tracks.items():
            if object_name == "ball" or object_name == "referees":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                if last_frame <= frame_num:
                    continue
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    start_position = object_tracks[frame_num][track_id].get('position_transformed', None)
                    end_position = object_tracks[last_frame][track_id].get('position_transformed', None)
                    if start_position is None or end_position is None:
                        continue
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed == 0:
                        continue
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6
                    
                    if object_name not in total_distance:
                        total_distance[object_name] = {}

                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0
                    total_distance[object_name][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if frame_num_batch < len(tracks[object_name]) and track_id in tracks[object_name][frame_num_batch]:
                            tracks[object_name][frame_num_batch][track_id]["speed"] = speed_km_per_hour
                            tracks[object_name][frame_num_batch][track_id]["total_distance"] = total_distance[object_name][track_id]
                    
    def draw_speed_and_distance(self, frames, tracks):
        single_frame = False
        if isinstance(frames, np.ndarray):
            frames = [frames]
            single_frame = True

        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name == "ball" or object_name == "referees":
                    continue
                if frame_num >= len(object_tracks):
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("total_distance", track_info.get("distance", None))
                        if speed is None or distance is None:
                            continue
                        bbox = track_info.get("bbox")
                        if bbox is None:
                            continue
                        position = get_foot_position(bbox)
                        pos_x, pos_y = int(position[0]), int(position[1] + 40)
                        cv2.putText(frame, f"{speed:.2f} km/h", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (pos_x, pos_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        if single_frame:
            return output_frames[0]
        return output_frames