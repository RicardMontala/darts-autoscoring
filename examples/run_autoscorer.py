import cv2
import numpy as np
from darts_autoscorer import (
    VideoInput,
    MotionCompensator,
    DartDetector,
    BoardCalibrator,
    ScoreCalculator,
)


# Parameters
video_path = "Video1.mp4"

# Open video
video = VideoInput(video_path)

# Initialise classes and states
compensator = MotionCompensator()
calibrator = BoardCalibrator(output_size=800, margin=40, radius_fraction=1.0)
score_calculator = ScoreCalculator(board_center=calibrator.board_center, board_radius=calibrator.board_radius)
detector = DartDetector(calibrator.board_center, warmup_frames=30*6, max_shift=5, frame_patience=3)

captured_vis = np.zeros((800, 800, 3), dtype=np.uint8)  # Shows only the latest frozen capture
capture_count = 0
scored_darts = []

# Read video frame by frame
for i in range(video.total_frames):
    frame = video.read_frame()
    if frame is None:
        break

    # Compensate video, align frame and find the new darts
    if i == 0:
        calibrator.calibrate(frame)

    frame, H = compensator.align(frame)
    frame = calibrator.rectify(frame)

    foreground_mask, new_dart_mask, boxes, tip = detector.detect(frame)
    mask_vis = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR) # Convert mask to 3-channel BGR just for visualization

    # Draw candidate bounding boxes
    for x, y, w, h in boxes:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Score newly detected dart
    if tip is not None:
        capture_count += 1

        score = score_calculator.calculate_score(tip)
        scored_darts.append({"tip": tip,"score": score})

        print(f"Dart tip: {tip} | Score: {score}")

        # Save frozen snapshot for visual verification
        captured_silhouette = new_dart_mask.copy()
        captured_vis = cv2.cvtColor(captured_silhouette, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(captured_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(captured_vis, f"Dart #{capture_count}", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Dart #{capture_count}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the scoring zones in the frame + the already scored darts
    frame = score_calculator.draw_scoring_zones(frame)

    for dart in scored_darts:
        tip = dart["tip"]
        score = dart["score"]
    
        cv2.circle(frame, tip, 6, (0, 0, 255), -1)
    
        cv2.putText(
            frame,
            str(score),
            (tip[0] + 10, tip[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Video", frame)
    cv2.imshow("New dart mask (live)", mask_vis)
    cv2.imshow("Last Captured dart", captured_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

video.close()
cv2.destroyAllWindows()