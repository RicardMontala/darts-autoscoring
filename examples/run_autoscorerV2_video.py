
import cv2
from darts_autoscorer import (
    VideoInput,
    MotionCompensator,
    DartDetector,
    BoardCalibrator,
    ScoreCalculator,
)

# Parameters
video_path = "Video1.mp4"
DISAPPEARANCE_PATIENCE = 3

# Open video
video = VideoInput(video_path)

# Initialise classes
compensator = MotionCompensator()
calibrator = BoardCalibrator(output_size=800, margin=40, radius_fraction=1.0)
detector = DartDetector(threshold=80, min_area=5)


# 1. INITIALISATION
# Read the first frame
previous_frame = video.read_frame()

if previous_frame is None:
    raise RuntimeError("Could not read the first frame.")

# Manually calibrate the board (only once)
calibrator.calibrate(previous_frame)

# Initialise the motion compensator
compensator.align(previous_frame)

# Initialise the score calculator
board_center = (
    calibrator.output_size // 2,
    calibrator.output_size // 2,
)

board_radius = calibrator.output_size / 2 - calibrator.margin

score_calculator = ScoreCalculator(
    board_center=board_center,
    board_radius=board_radius,
)


# 2. INITIALIZE DETECTION STATE
dart_detected = False   # There is a dart in the frame
last_detection = None   # Info of the last dart detected
missing_frames = 0      # Number of frames where the dart has dissapear

scored_darts = []       # Store scored darts


# 3. PROCESS THE REST OF THE VIDEO
while True:

    current_frame = video.read_frame()

    if current_frame is None:
        break

    # 1. Motion compensation
    aligned_current, H = compensator.align(current_frame)

    if H is None:
        print("Alignment failed. Using original frame.")
        aligned_current = current_frame

    # 2. Rectify both frames
    rectified_reference = calibrator.rectify(previous_frame)
    rectified_current = calibrator.rectify(aligned_current)

    # 3. Detect newly appearing darts
    mask, boxes = detector.detect(
        rectified_reference,
        rectified_current,
    )

    # 4. Update detection state
    if boxes:

        # A dart is currently detected
        dart_detected = True
        missing_frames = 0

        # Keep the last frame and detection information
        last_detection = {
            "frame": rectified_current.copy(),
            "mask": mask.copy(),
            "bbox": max(boxes, key=lambda b: b[2] * b[3]),
        }

    elif dart_detected:

        # The dart was previously detected but is now missing
        missing_frames += 1

        if missing_frames >= DISAPPEARANCE_PATIENCE:

            # 5. Locate dart tip using the last detected mask
            tip = detector.locate_tip(
                last_detection["mask"],
                last_detection["bbox"],
                board_center,
            )

            if tip is not None:

                # Calculate score
                score = score_calculator.calculate_score(tip)
                scored_darts.append({"tip": tip,"score": score})

                print(f"Dart tip: {tip} | Score: {score}")

            # Reset detection state
            dart_detected = False
            missing_frames = 0
            last_detection = None


    # 6. Display current frame and previously scored darts
    debug_darts = rectified_current.copy()
    debug_darts = score_calculator.draw_scoring_zones(debug_darts)

    for dart in scored_darts:

        tip = dart["tip"]
        score = dart["score"]

        cv2.circle(debug_darts, tip, 6, (0, 0, 255), -1)

        cv2.putText(
            debug_darts,
            str(score),
            (tip[0] + 10, tip[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Detected dart tips", debug_darts)
    cv2.imshow("Frame differences", mask)

    key = cv2.waitKey(5) & 0xFF

    if key == ord("q") or key == 27:
        break

    # 7. Update reference frame
    previous_frame = aligned_current


# 3. CLEAN UP
video.close()
cv2.destroyAllWindows()
