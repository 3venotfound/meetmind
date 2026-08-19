"""
make_synthetic_test_video.py

This is NOT part of the CV pipeline itself. It's a helper script that
creates a fake short video so we can test video_processor.py without
needing a real meeting recording yet.

The video is 10 seconds long at 10 fps. Each frame just shows a big
number counting up the elapsed seconds, e.g. "0", "1", "2" ... "9",
so when we look at the extracted frames afterward we can immediately
tell, just by eye, whether extract_frames() grabbed frames at the
correct points in time.
"""

import cv2
import numpy as np

OUTPUT_PATH = "input/meeting_001.mp4"
WIDTH, HEIGHT = 640, 480
FPS = 10
DURATION_SECONDS = 10

# fourcc is the video codec identifier. "mp4v" is a widely supported
# codec for writing .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

for frame_index in range(FPS * DURATION_SECONDS):
    elapsed_seconds = frame_index / FPS

    # Start with a plain dark gray background.
    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)

    # Draw the current elapsed time as large white text in the middle.
    text = f"{elapsed_seconds:.1f}s"
    cv2.putText(
        frame, text, (150, 260),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4
    )

    writer.write(frame)

writer.release()
print(f"Synthetic test video written to {OUTPUT_PATH} ({DURATION_SECONDS}s @ {FPS}fps)")
