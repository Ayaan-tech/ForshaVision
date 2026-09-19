import pandas as pd
import numpy as np
import os
import cv2
import pickle
import sys
import supervision as sv
from ultralytics import YOLO
sys.path.append('../')
from training.utils import get_bbox_width, get_center_bbox, get_foot_position
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
                    
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def process_video_stream(self, frame_generator, batch_size=4, read_from_stub=False, stub_path=None):
        """
        Processes a streaming generator of frames or instantly loads data from a stub.
        This handles the high-level loading state cleanly.
        """
        # 1. Check for the stub at the entry point BEFORE reading any frames
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading tracking data from existing stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                self.tracks = pickle.load(f)
            return self.tracks

        print("No stub found or read_from_stub=False. Running YOLO inference...")
        batch = []
        frame_num = 0

        for frame in frame_generator:
            batch.append(frame)

            # Once the batch is full, process it
            if len(batch) == batch_size:
                self._process_batch(batch, frame_num)
                frame_num += len(batch)
                batch = []  # Clear the batch from memory instantly

        # Process any remaining frames at the end of the video
        if batch:
            self._process_batch(batch, frame_num)

        # 2. Save the stub ONCE after the entire video is finished processing
        if stub_path is not None:
            output_dir = os.path.dirname(stub_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(self.tracks, f)
            print(f"Tracking data saved successfully to stub: {stub_path}")

        return self.tracks

    def _process_batch(self, batch, start_frame_idx):
        """Helper to run inference and safely map tracked components on a concrete frame list."""
        detections_batch = self.model.predict(batch, conf=0.1, verbose=False)
        
        for idx, detection in enumerate(detections_batch):
            frame_num = start_frame_idx + idx
            cls_names = detection.names
            cls_names_inverse = {value: key for key, value in cls_names.items()}
            
            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert Goalkeeper to player object safely
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inverse["player"]
            
            # Track Objects using byte track
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialize empty structures for the current frame index
            frame_players = {}
            frame_referees = {}
            frame_ball = {}

            # Process tracked objects safely using supervision attributes
            if detection_with_tracks.tracker_id is not None:
                for box, class_id, track_id in zip(
                    detection_with_tracks.xyxy, 
                    detection_with_tracks.class_id, 
                    detection_with_tracks.tracker_id
                ):
                    bbox = box.tolist()
                    if class_id == cls_names_inverse["player"]:
                        frame_players[int(track_id)] = {"bbox": bbox}
                    elif class_id == cls_names_inverse["referee"]:
                        frame_referees[int(track_id)] = {"bbox": bbox}

            # Process non-tracked elements (the football/soccer ball)
            for box, class_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                if class_id == cls_names_inverse["ball"]:
                    frame_ball[1] = {"bbox": box.tolist()}

            # Append the structured frame mappings to the primary tracked sequence lists
            self.tracks["players"].append(frame_players)
            self.tracks["referees"].append(frame_referees)
            self.tracks["ball"].append(frame_ball)
            
            print(f"Processed Frame {frame_num} | Players: {len(frame_players)} | Referees: {len(frame_referees)}")
    def draw_ellipse(self,frame,bbox,color,track_id):
        y2 = int(bbox[3])
        x_center,_ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rect_width = 40
        rect_height=  20
        x1_rect = x_center - rect_width//2
        x2_rect = x_center + rect_width//2
        y1_rect = (y2 - rect_height//2)+15
        y2_rect = (y2 + rect_height//2)+15
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+5)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0 ),
                2
            )
        return frame
    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_ = get_center_bbox(bbox)
        triangle_points = np.array([
            [x,y-10],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi transparent rectangle
        overlay = frame[850:970, 1350:1900].copy()
        white_rect = np.ones_like(overlay) * 255
        alpha = 0.4
        cv2.addWeighted(white_rect, alpha, overlay, 1 - alpha, 0, overlay)
        frame[850:970, 1350:1900] = overlay
        
        team_ball_control_till_frame = np.array(team_ball_control)
        # Get the number of time each team had ball control
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total_frames if total_frames > 0 else 0.0
        team_2 = team_2_num_frames / total_frames if total_frames > 0 else 0.0
        cv2.putText(
            frame,
            f"Team 1: {team_1:.2%}",
            (1400,900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            3
        )
        cv2.putText(
            frame,
            f"Team 2: {team_2:.2%}",
            (1400,930),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            3
        )
        return frame
    def draw_annotation(self, video_Frames, tracks, team_ball_control):
        output_Video_frames = []
        for frame_num, frame in enumerate(video_Frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # Draw players with ellipse
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball",False):
                    frame = self.draw_triangle(frame,player["bbox"],color)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), None)
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            # Draw Team ball control
            control_history = team_ball_control[:frame_num + 1] if len(team_ball_control) == len(video_Frames) else team_ball_control
            frame = self.draw_team_ball_control(frame, frame_num, control_history)
            output_Video_frames.append(frame)   
        return output_Video_frames 

            