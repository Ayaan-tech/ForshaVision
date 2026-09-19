
from training.utils import measure_distance, measure_xy_distance

import cv2
import pickle
import os
import numpy as np


class CameraMovementEstimator:

    def __init__(self, frame):

        self.first_frame_grayscale = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

        mask_features = np.zeros_like(
            self.first_frame_grayscale
        )

        # Your original video is 1920x1080,
        # so these coordinates are valid.
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1500] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        # ========================================================
        # Load existing stub if available
        # ========================================================

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):

            print(
                f"Loading camera movement from stub: {stub_path}"
            )

            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)

            return camera_movement

        print(
            "No camera movement stub found. "
            "Calculating camera movement..."
        )

        # ========================================================
        # Convert generator into an iterator
        # ========================================================

        frame_iterator = iter(frames)

        # Get first frame
        try:
            first_frame = next(frame_iterator)

        except StopIteration:

            raise ValueError(
                "Video contains no frames."
            )

        # ========================================================
        # Initialize
        # ========================================================

        camera_movement = [[0, 0]]

        old_gray = cv2.cvtColor(
            first_frame,
            cv2.COLOR_BGR2GRAY
        )

        old_features = cv2.goodFeaturesToTrack(
            old_gray,
            **self.features
        )

        # ========================================================
        # Process remaining frames one by one
        # ========================================================

        for frame_num, frame in enumerate(
            frame_iterator,
            start=1
        ):

            frame_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            camera_movement_x = 0
            camera_movement_y = 0
            max_distance = 0

            # ----------------------------------------------------
            # If no features were detected, find new ones
            # ----------------------------------------------------

            if old_features is None:

                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )

            # ----------------------------------------------------
            # Optical flow
            # ----------------------------------------------------

            if old_features is not None:

                new_features, status, _ = (
                    cv2.calcOpticalFlowPyrLK(
                        old_gray,
                        frame_gray,
                        old_features,
                        None,
                        **self.lk_params
                    )
                )

                if (
                    new_features is not None
                    and status is not None
                ):

                    for i, (new, old) in enumerate(
                        zip(new_features, old_features)
                    ):

                        if status[i][0] == 0:
                            continue

                        new_features_point = new.ravel()
                        old_features_point = old.ravel()

                        distance = measure_distance(
                            new_features_point,
                            old_features_point
                        )

                        if distance > max_distance:

                            max_distance = distance

                            (
                                camera_movement_x,
                                camera_movement_y
                            ) = measure_xy_distance(
                                new_features_point,
                                old_features_point
                            )

            # ----------------------------------------------------
            # Decide whether camera moved
            # ----------------------------------------------------

            if max_distance > self.minimum_distance:

                camera_movement.append([
                    camera_movement_x,
                    camera_movement_y
                ])

                # Re-detect features
                old_features = cv2.goodFeaturesToTrack(
                    frame_gray,
                    **self.features
                )

            else:

                camera_movement.append([
                    0,
                    0
                ])

            # ----------------------------------------------------
            # Current frame becomes previous frame
            # ----------------------------------------------------

            old_gray = frame_gray

        # ========================================================
        # Save stub
        # ========================================================

        if stub_path is not None:

            output_dir = os.path.dirname(
                stub_path
            )

            if output_dir:
                os.makedirs(
                    output_dir,
                    exist_ok=True
                )

            with open(stub_path, 'wb') as f:

                pickle.dump(
                    camera_movement,
                    f
                )

            print(
                f"Camera movement saved to: {stub_path}"
            )

        return camera_movement

    def add_adjust_positions_to_tracks(
        self,
        tracks,
        camera_movement_per_frame
    ):

        for object_name, object_tracks in tracks.items():

            for frame_num, track in enumerate(
                object_tracks
            ):

                # Safety check
                if frame_num >= len(
                    camera_movement_per_frame
                ):
                    continue

                for track_id, track_info in track.items():

                    position = track_info.get(
                        'position'
                    )

                    if position is None:
                        continue

                    camera_movement = (
                        camera_movement_per_frame[
                            frame_num
                        ]
                    )

                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )

                    tracks[
                        object_name
                    ][
                        frame_num
                    ][
                        track_id
                    ][
                        'position_adjusted'
                    ] = position_adjusted

        return tracks

    def _adjust_positions_to_track(
        self,
        tracks,
        camera_movement_per_frame
    ):

        return self.add_adjust_positions_to_tracks(
            tracks,
            camera_movement_per_frame
        )

    def draw_camera_movement(
        self,
        frames,
        camera_movement_per_frame
    ):

        output_frames = []

        for frame_num, frame in enumerate(frames):

            frame = frame.copy()

            overlay = frame[
                0:100,
                0:500
            ].copy()

            white_rect = np.ones_like(
                overlay
            ) * 255

            alpha = 0.6

            cv2.addWeighted(
                white_rect,
                alpha,
                overlay,
                1 - alpha,
                0,
                overlay
            )

            frame[
                0:100,
                0:500
            ] = overlay

            x_movement, y_movement = (
                camera_movement_per_frame[
                    frame_num
                ]
            )

            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            output_frames.append(frame)

        return output_frames

