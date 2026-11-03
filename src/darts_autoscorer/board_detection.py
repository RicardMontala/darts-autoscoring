import cv2
import numpy as np


class BoardDetector:
    """
    Detect a dartboard using OpenCV edge and contour analysis.

    Returns the best approximately circular candidate without requiring
    manually specified crop coordinates.
    """

    def __init__(
        self,
        min_diameter_fraction=0.15,
        max_diameter_fraction=0.95,
        min_aspect_ratio=0.75,
        min_circularity=0.40,
        blur_kernel=5,
        canny_threshold1=40,
        canny_threshold2=120,
        close_kernel_size=5,
    ):
        self.min_diameter_fraction = min_diameter_fraction
        self.max_diameter_fraction = max_diameter_fraction
        self.min_aspect_ratio = min_aspect_ratio
        self.min_circularity = min_circularity
        self.blur_kernel = blur_kernel
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.close_kernel_size = close_kernel_size


    def detect(self, frame):
        """
        Parameters
        ----------
        frame: np.ndarray
            Input BGR image.

        Returns
        -------
        board: dict or None
            Center, ellipse axes, angle, and bounding box.
            None if no suitable candidate is found.

        debug: np.ndarray
            Image showing the detected candidate and its center.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        height, width = frame.shape[:2]
        image_size = min(height, width)

        # Convert to gray-scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur filter (noise reduction)
        kernel = self.blur_kernel
        if kernel % 2 == 0:
            kernel += 1

        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        # Detect edges (where images intensity change abruptly): Dardboard limits
        # (Creates an image of 0s or 255s, where 255 indicates the location of edges)
        edges = cv2.Canny(
            blurred,
            self.canny_threshold1,
            self.canny_threshold2,
        )

        # Join nearby edge fragments
        # (if a closed patters arise with some gaps, it fills those gaps)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.close_kernel_size, self.close_kernel_size),
        )

        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        # Detect contours. This is a list of contours, where each contour has 
        # a shape of (N, 1, 2); where N is the number of stored points and each point
        # contains (x, y) coordinates 
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST, # retains inner contours as well as outer contours
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Evaluate obtained contours
        candidates = []

        min_diameter = image_size * self.min_diameter_fraction
        max_diameter = image_size * self.max_diameter_fraction

        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0 or len(contour) < 5:
                continue

            area = cv2.contourArea(contour)

            if area <= 0:
                continue

            # Circularity is 1 for an ideal circle
            circularity = 4.0 * np.pi * area / (perimeter ** 2)

            if circularity < self.min_circularity:
                continue

            # Fit elipse
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis1, axis2), angle = ellipse

            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if minor_axis <= 0:
                continue

            if not min_diameter <= major_axis <= max_diameter:
                continue

            # In a circle, minor_axis = major_axis, so the aspect ratio =1
            aspect_ratio = minor_axis / major_axis

            if aspect_ratio < self.min_aspect_ratio:
                continue

            # Reject ellipses extending substantially beyond the image.
            if (
                cx - major_axis / 2 < -0.05 * width
                or cx + major_axis / 2 > 1.05 * width
                or cy - major_axis / 2 < -0.05 * height
                or cy + major_axis / 2 > 1.05 * height
            ):
                continue

            # Distance from the image center. Probably the board will be close to the center
            center_distance = np.sqrt(
                ((cx - width / 2) / width) ** 2
                + ((cy - height / 2) / height) ** 2
            )

            # We want to take the largest candidate relative to the image size
            # The board will probably be the largest circular object in the image
            size_score = major_axis / image_size

            # Compute total score
            score = (
                size_score          # We want this to be 1
                * circularity       # We want this to be 1
                * aspect_ratio      # We want this to be 1
                / (1.0 + 2.0 * center_distance) # We want center_distance to be 0
            )

            candidates.append({
                "center": (float(cx), float(cy)),
                "axes": (float(axis1), float(axis2)),
                "angle": float(angle),
                "bbox": cv2.boundingRect(contour),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "score": float(score),
            })

        # Select the highest-scoring candidate
        debug = frame.copy()

        if not candidates:
            return None, debug

        board = max(candidates, key=lambda item: item["score"])

        center = tuple(np.round(board["center"]).astype(int))
        axes = tuple(max(1, int(round(axis / 2))) for axis in board["axes"])

        cv2.ellipse(
            debug,
            center,
            axes,
            board["angle"],
            0,
            360,
            (0, 255, 0),
            3,
        )

        cv2.circle(debug, center, 5, (0, 0, 255), -1)

        x, y, w, h = board["bbox"]
        cv2.rectangle(
            debug,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            debug,
            f"Score: {board['score']:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return board, debug


    def crop_board(self, frame, board, mask_margin=0.80):
        """
        Extract the board region using a circular mask.

        Returns
        -------
        crop : np.ndarray
            Rectangular image containing the circular board region.
            Pixels outside the circle are black.
        """

        if board is None:
            raise ValueError("No board object was provided.")

        height, width = frame.shape[:2]

        cx, cy = board["center"]
        axis1, axis2 = board["axes"]

        # Define the circular ROI in the original image
        radius = int(0.5 * min(axis1, axis2) * mask_margin)

        # Calculate bounding square of the circle
        x1 = max(0, int(cx - radius))
        y1 = max(0, int(cy - radius))
        x2 = min(width, int(cx + radius))
        y2 = min(height, int(cy + radius))

        # Crop the image
        crop = frame[y1:y2, x1:x2].copy()

        # Circle center relative to the crop
        crop_center = (int(round(cx - x1)), int(round(cy - y1)))

        # Create and apply circular mask
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)

        cv2.circle(
            mask,
            crop_center,
            radius,
            255,
            thickness=-1,
        )

        crop = cv2.bitwise_and(crop, crop, mask=mask)

        return crop

