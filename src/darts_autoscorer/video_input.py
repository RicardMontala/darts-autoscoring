
from pathlib import Path

import cv2
import numpy as np


class VideoInput:
    '''
    Reads and ispects a video using OpenCV.
    '''

    def __init__(self, video_path: str):

        # Read the video
        self.cap = cv2.VideoCapture(video_path)

        # Set video properties
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Number of [height, width] pixels
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # Frames per second
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames

        self.duration = self.total_frames / self.fps # Video duration in seconds


    def read_frame(self) -> np.ndarray | None:
        '''
        At every call, reads the next frame starting from frame 0.

        Returns BGR image if successful, otherwise None at the end of file.
        '''

        success, frame = self.cap.read()

        if not success:
            return None

        return frame


    def seek (self, frame_number: int) -> bool:
        '''
        Move to a specific frame given its index (frame_number)
        '''

        if not 0 <= frame_number < self.total_frames:
            raise ValueError(f"Frame index must be between 0 and {self.total_frames - 1}.")

        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


    def reset(self) -> None:
        '''
        Return to the beginning of the video.
        '''

        self.seek(0)


    def show_frame(self, frame_number:int, window_name: str = "Frame") -> None:
        '''
        Show a specific frame given its index (frame_number).
        '''

        self.seek(frame_number)
        frame = self.read_frame()

        cv2.imshow(window_name, frame)
        cv2.waitKey(0) # waits indefinitely until you press a key

        cv2.destroyWindow(window_name)


    def show_video(self, window_name: str = "Video") -> None:
        '''
        Play the video frame by frame in an OpenCV window.
        '''

        self.reset()

        while True:
            frame = self.read_frame()

            if frame is None:
                break

            cv2.imshow(window_name, frame)

            # Press q or ESC to exit playback.
            key = cv2.waitKey(max(1, round(1000 / self.fps))) & 0xFF

            if key == ord("q") or key == 27:
                break

        cv2.destroyWindow(window_name)


    def close(self) -> None:
        '''
        Release the video capture resource.
        '''

        if self.cap is not None:
            self.cap.release()
            self.cap = None
