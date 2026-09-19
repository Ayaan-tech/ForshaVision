from training.speed_and_distance_estimator import SpeedAndDistanceEstimator
import os
from training.utils import save_Video
from training.utils import read_video

import numpy as np

from training.movement import CameraMovementEstimator
from training.tracker.tracking import Tracker
from training.team_assigner import TeamAssigner
from training.perspective_transformer import ViewTransformer
from training.player_ball_assigner import PlayerBallAssigner

import cv2


def get_first_frame(video_path):
    """
    Read only the first frame from the video.
    """
    video = cv2.VideoCapture(video_path)

    ret, frame = video.read()

    video.release()

    if not ret:
        raise ValueError(f"Could not read video: {video_path}")

    return frame


def main():

    video_path = 'input_videos/input_1.mp4'

    # ============================================================
    # 1. Initialize tracker
    # ============================================================

    tracker = Tracker(
        'training/models/best.pt'
    )

    # ============================================================
    # 2. Tracking
    #
    # read_video() creates a fresh generator.
    # The entire video is NOT loaded into RAM.
    # ============================================================

    tracks = tracker.process_video_stream(
        frame_generator=read_video(video_path),
        batch_size=4,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Add player/ball positions
    tracker.add_position_to_tracks(tracks)

    # ============================================================
    # 3. Camera Movement
    # ============================================================

    first_frame = get_first_frame(video_path)

    camera_movement_estimator = CameraMovementEstimator(
        first_frame
    )

    camera_movement_per_frame = (
        camera_movement_estimator.get_camera_movement(
            read_video(video_path),
            read_from_stub=True,
            stub_path='stubs/camera_movement_stub.pkl'
        )
    )

    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks,
        camera_movement_per_frame
    )

    # ============================================================
    # 4. Perspective Transformation
    # ============================================================

    view_transformer = ViewTransformer()

    view_transformer.add_transformed_position_to_track(
        tracks
    )

    # ============================================================
    # 5. Interpolate Ball Positions
    # ============================================================

    tracks["ball"] = tracker.interpolate_ball_positions(
        tracks["ball"]
    )
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ============================================================
    # 6. Team Assignment
    # ============================================================

    team_assigner = TeamAssigner()

    # Read first frame again
    first_frame = get_first_frame(video_path)

    team_assigner.assign_team_color(
        first_frame,
        tracks["players"][0]
    )

    # Read video sequentially for team assignment
    for frame_num, frame in enumerate(
        read_video(video_path)
    ):

        if frame_num >= len(tracks["players"]):
            break

        player_track = tracks["players"][frame_num]

        for player_id, track in player_track.items():

            team = team_assigner.get_player_team(
                frame,
                track["bbox"],
                player_id
            )

            tracks["players"][frame_num][player_id]["team"] = team

            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # ============================================================
    # 7. Ball Possession with Temporal Smoothing
    # ============================================================

    player_ball_assigner = PlayerBallAssigner()

    team_ball_control = []
    last_team_control = -1

    for frame_num, player_track in enumerate(
        tracks["players"]
    ):
        ball_data = tracks["ball"][frame_num].get(1)
        assigned_player = -1
        if ball_data is not None:
            ball_bbox = ball_data.get("bbox")
            if ball_bbox is not None:
                assigned_player = (
                    player_ball_assigner.assign_ball_to_player(
                        player_track,
                        ball_bbox
                    )
                )

        if assigned_player != -1:
            tracks["players"][
                frame_num
            ][assigned_player]["has_ball"] = True

            team = tracks["players"][
                frame_num
            ][assigned_player]["team"]
            team_ball_control.append(team)
            last_team_control = team
        else:
            # Temporal smoothing: carry forward last known team possession
            team_ball_control.append(last_team_control)

    team_ball_control = np.array(
        team_ball_control
    )

    # ============================================================
    # 8. Draw annotations
    #
    # IMPORTANT:
    # We do NOT create video_frames = [...]
    #
    # Instead, process frames one at a time.
    # ============================================================

    output_video_path = (
        "output_videos/tracked_video.mp4"
    )

    os.makedirs(
        "output_videos",
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    # Get video properties
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if fps <= 0:
        fps = 24.0

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    # Process frames one at a time
    for frame_num, frame in enumerate(
        read_video(video_path)
    ):

        if frame_num >= len(
            tracks["players"]
        ):
            break

        # --------------------------------------------------------
        # Draw tracking annotations
        # --------------------------------------------------------

        annotated_frame = tracker.draw_annotation(
            [frame],
            {
                "players": [
                    tracks["players"][frame_num]
                ],
                "ball": [
                    tracks["ball"][frame_num]
                ],
                "referees": [
                    tracks["referees"][frame_num]
                ]
            },
            team_ball_control[
                :frame_num + 1
            ]
        )[0]

        # --------------------------------------------------------
        # Camera movement annotation
        # --------------------------------------------------------

        annotated_frame = camera_movement_estimator.draw_camera_movement(
            [annotated_frame],
            camera_movement_per_frame[
                frame_num:frame_num + 1
            ]
        )[0]
        annotated_frame = speed_and_distance_estimator.draw_speed_and_distance(
            [annotated_frame],
            {
                "players": [
                    tracks["players"][frame_num]
                ]
            }
        )[0]

        out.write(
            annotated_frame
        )

    out.release()

    print(
        f"Output video saved to: {output_video_path}"
    )


if __name__ == "__main__":
    main()

