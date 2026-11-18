import cv2
import numpy as np


class BoardCalibrator:
    """
    Rectify an approximately frontal dartboard using its detected ellipse.

    The ellipse is transformed into a circle using an affine transformation.
    The resulting board is centred in a square image.
    """

    def __init__(self, output_size=800, margin=40, radius_fraction=0.8):
        self.output_size = output_size
        self.margin = margin # Pixels distance from the outer circle and the image limits
        self.radius_fraction = radius_fraction

        self.H = None
        self.source_points = None
        self.destination_points = None

        self.board_center = (self.output_size // 2, self.output_size // 2)
        self.board_radius = self.output_size / 2 - self.margin

    def calibrate(self, frame):

        # Select four points: top, right, bottom, left
        points = []

        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])

        display = frame.copy()
        window_name = "Select board points"

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_point)

        while len(points) < 4:

            display = frame.copy()

            for i, point in enumerate(points):
                cv2.circle(display, tuple(point), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    str(i + 1),
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                raise RuntimeError("Calibration cancelled.")

        cv2.destroyWindow(window_name)

        self.source_points = np.array(
            points,
            dtype=np.float32
        )

        center = self.output_size / 2
        radius = center - self.margin

        self.destination_points = np.array([
            [center, center - radius],  # Top
            [center + radius, center],  # Right
            [center, center + radius],  # Bottom
            [center - radius, center],  # Left
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(
            self.source_points,
            self.destination_points
        )

        return self.H


    def rectify(self, frame):

        if self.H is None:
            raise RuntimeError("Call calibrate() first.")

        # Apply perspective transformation
        rectified = cv2.warpPerspective(
            frame,
            self.H,
            (self.output_size, self.output_size)
        )

        # Create circular mask
        mask = np.zeros(
            (self.output_size, self.output_size),
            dtype=np.uint8,
        )

        center = (self.output_size // 2, self.output_size // 2)
        max_radius = self.output_size / 2
        radius = int(max_radius * self.radius_fraction)

        cv2.circle(
            mask,
            center,
            radius,
            255,
            thickness=-1,
        )

        # Apply mask
        rectified = cv2.bitwise_and(
            rectified,
            rectified,
            mask=mask,
        )

        return rectified