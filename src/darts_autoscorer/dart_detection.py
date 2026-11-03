
import cv2
import numpy as np


class DartDetector:
    """
    Detect changed image regions between two board frames.
    """

    def __init__(self, threshold=25, min_area=15):
        self.threshold = threshold
        self.min_area = min_area


    def detect(self, reference_frame, current_frame):
        """
        Compare a reference frame with a current frame.

        Returns
        -------
        mask: np.ndarray
            Binary mask of detected changes.
        boxes: list[tuple[int, int, int, int]]
            Bounding boxes (x, y, width, height) of candidate regions.
        """

        if reference_frame.shape != current_frame.shape:
            raise ValueError("Both frames must have the same dimensions.")

        # Convert to grayscale
        reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Reduce small image noise before comparing
        #reference_gray = cv2.GaussianBlur(reference_gray, (3, 3), 0)
        #current_gray = cv2.GaussianBlur(current_gray, (3, 3), 0)

        # Compute absolute pixel-wise differences
        difference = cv2.absdiff(reference_gray, current_gray)

        # Keep pixels whose intensity changed sufficiently
        _, mask = cv2.threshold(
            difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )

        # Connect nearby changed pixels without aggressively
        # removing the thin structures of a dart.
        #kernel = np.ones((2, 2), np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (12, 12),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find connected changed regions
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            boxes.append((x, y, width, height))

        return mask, boxes


    def locate_tip(self, mask, bbox, board_center):
        """
        Estimate the dart tip using PCA.

        Parameters
        ----------
        mask : np.ndarray
            Binary difference mask returned by detect().
        bbox : tuple
            Single bounding box (x, y, width, height) returnned by detect().
        board_center : tuple
            Centre of the rectified dartboard.

        Returns
        -------
        tuple[int, int] or None
            Estimated tip coordinates in the full image.
        """

        x, y, w, h = bbox

        # Extract the dart region.
        roi = mask[y:y+h, x:x+w]

        # Find the contour.
        contours, _ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return None

        # Pick the contour with the biggest area
        contour = max(contours, key=cv2.contourArea)

        # Reshape from (N,1,2) (given by OpenCV) to (N,2) -> [[x,y]] of contour points
        points = contour.reshape(-1, 2).astype(np.float32)

        # At least we need 3 points to perform a meaningfully PCA
        if len(points) < 3:
            return None

        # Calculate the principal axis using PCA.
        mean, eigenvectors = cv2.PCACompute(points, mean=None)

        # Project points onto the principal axis.
        axis = eigenvectors[0] # Principal axis (where the dart tip is located)
        projections = (points - mean) @ axis

        # Find the two extreme points.
        point_a = points[np.argmin(projections)]
        point_b = points[np.argmax(projections)]

        # Convert ROI coordinates to full-image coordinates.
        point_a += np.array([x, y])
        point_b += np.array([x, y])

        # Select the endpoint closest to the board centre.
        board_center = np.array(board_center)

        dist_a = np.linalg.norm(point_a - board_center)
        dist_b = np.linalg.norm(point_b - board_center)

        tip = point_a if dist_a < dist_b else point_b

        return tuple(np.round(tip).astype(int))

