import cv2
import numpy as np


class ScoreCalculator:
    """
    Calculate dart scores from tip coordinates on a rectified dartboard.
    """

    SECTORS = [
        20, 1, 18, 4, 13,
        6, 10, 15, 2, 17,
        3, 19, 7, 16, 8,
        11, 14, 9, 12, 5
    ]

    def __init__(self, board_center, board_radius):
        self.board_center = np.array(board_center, dtype=float)

        # Convert standard dartboard dimensions from mm to pixels.
        scale = board_radius / 170.0

        self.r_inner_bull = 6.35 * scale
        self.r_outer_bull = 15.9 * scale
        self.r_inner_treble = 99.0 * scale
        self.r_outer_treble = 107.0 * scale
        self.r_inner_double = 162.0 * scale
        self.r_outer_double = 170.0 * scale

    def calculate_score(self, tip):
        """
        Calculate the score of a dart from its tip coordinates.

        Parameters
        ----------
        tip : tuple[int, int]
            Dart tip coordinates (x, y).

        Returns
        -------
        int
            Dart score.
        """

        dx = tip[0] - self.board_center[0]
        dy = tip[1] - self.board_center[1]

        # Radial distance from the bullseye.
        r = np.sqrt(dx**2 + dy**2)

        # Bullseye and outer bull.
        if r <= self.r_inner_bull:
            return 50

        if r <= self.r_outer_bull:
            return 25

        # Outside the scoring area.
        if r > self.r_outer_double:
            return 0

        # Determine the angular sector.
        angle = np.degrees(np.arctan2(dx, -dy)) % 360 # Top is 0º an angle increases clockwise
        sector_index = int((angle + 9) // 18) % 20 # Each sector spans 18º
        sector = self.SECTORS[sector_index]

        # Determine the multiplier.
        if self.r_inner_treble <= r <= self.r_outer_treble:
            multiplier = 3
        elif self.r_inner_double <= r <= self.r_outer_double:
            multiplier = 2
        else:
            multiplier = 1

        return sector * multiplier


    def draw_scoring_zones(self, image):
        """
        Draw the scoring-zone boundaries on a rectified dartboard image
        (after aplying calibration).
        """

        center = tuple(self.board_center.astype(int))

        # Draw radial boundaries
        for i in range(20):

            angle = np.radians(i * 18 - 9)

            # Direction vector
            dx = np.sin(angle)
            dy = -np.cos(angle)

            # Draw lines from the inner bull to the outer double boundary
            start = (
                int(center[0] + self.r_inner_bull * dx),
                int(center[1] + self.r_inner_bull * dy),
            )

            end = (
                int(center[0] + self.r_outer_double * dx),
                int(center[1] + self.r_outer_double * dy),
            )

            cv2.line(image, start, end, (0, 255, 0), 1)

        # Draw circular scoring boundaries
        radii = [
            self.r_inner_bull,
            self.r_outer_bull,
            self.r_inner_treble,
            self.r_outer_treble,
            self.r_inner_double,
            self.r_outer_double,
        ]

        for radius in radii:
            cv2.circle(
                image,
                center,
                int(round(radius)),
                (0, 255, 0),
                1,
            )

        return image
