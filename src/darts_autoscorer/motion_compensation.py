import cv2
import numpy as np

class MotionCompensator:
    """
    Align consecutive frames using ORB features and a homography.
    
    The previous frame becomes the refence for the next frame.
    
    """

    def __init__(self, max_features=2000):

        # Initialize Oriented FAST and Rotated BRIEF algorithm (Feature detection method)
        # Provides Keypoints and Descriptors to describe local image around the keypoints
        self.orb = cv2.ORB_create(nfeatures=max_features)

        # Create the matcher to compare ORB descriptors between frames
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

        # We compare gray intensities so we transform the image in grayscale before
        self.previous_gray = None


    def reset(self):
        """Start a new sequence of frames."""

        self.previous_gray = None


    def align(self, frame):
        """
        Align the current frame to the previous frame.

        Returns
        -------
        aligned_frame: ndarray
            Current frame warped into the previous frame's coordinates.

        H: ndarray or None
            Estimated homography mapping current to previous coordinates.
        """

        # Transform frame (RGB format) into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Handling the first frame...
        if self.previous_gray is None:
            self.previous_gray = gray
            return frame.copy(), None

        # Preserve the previous frame and advance the reference
        previous_gray = self.previous_gray
        self.previous_gray = gray

        # Handling the remaining frames...
        # Compute key points (kp) and descirptors (des) for each frame (previous and current)
        kp_prev, des_prev = self.orb.detectAndCompute(previous_gray, None)
        kp_curr, des_curr = self.orb.detectAndCompute(gray, None)

        # Check whether ORB sucessfully detected features in both frames
        if des_prev is None or des_curr is None:
            return frame.copy(), None

        # Compare descriptors between frames and obtain matches
        matches = self.matcher.match(des_curr, des_prev)
        matches = sorted(matches, key=lambda match: match.distance)

        # Keep the best matches
        matches = matches[:300]

        if len(matches) < 8:
            self.previous_gray = gray
            return frame.copy(), None

        # Extract matched pixels coordinates
        # e.g. Feature A: points_curr (415,307) and points_prev (400, 300)
        # OpenCV expects a shape of (N,1,2), where N is the number of matches and 2 corresponds to (x,y)
        points_curr = np.float32([kp_curr[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_prev = np.float32([kp_prev[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Obtain mapping (homography, H) from one image to the other (Change of basis matrix))
        # [x',y',w'] = H [x,y,1], where H is a 3x3 matrix
        # mask indicating which points are inliers and which are outliers
        H, mask = cv2.findHomography(points_curr, points_prev, cv2.RANSAC, 3.0,)

        # Check that the homoghraphy is reliable
        if H is None or mask is None:
            return frame.copy(), None

        # Apply homography to express the current color frame with the previous
        # frame's coordinate system
        height, width = frame.shape[:2]
        aligned = cv2.warpPerspective(frame, H, (width, height))

        return aligned, H