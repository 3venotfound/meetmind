"""
video_processor.py

PURPOSE
-------
This is the very first step of the MeetMind AI Computer Vision pipeline.

Given a meeting video file (e.g. "meeting_001.mp4"), this module:
    1. Opens the video
    2. Figures out its frame rate (fps) and duration
    3. "Samples" one frame every N seconds (instead of every single frame,
       which would be way too many images to process)
    4. Saves each sampled frame as a .jpg file to disk
    5. Returns a list telling us WHERE each frame is and WHEN in the
       video it was taken

WHY WE SAMPLE INSTEAD OF USING EVERY FRAME
-------------------------------------------
A video usually has 24-30 frames PER SECOND. A 10-minute meeting video
could have 15,000+ frames. We don't need that many - a whiteboard or
slide doesn't change every fraction of a second. So we only grab one
frame every `interval_seconds` (2 seconds by default) to keep things
fast and manageable. Later pipeline steps (detection, OCR) will run on
this much smaller set of sampled frames instead of the whole video.

This file only does ONE job: turn a video into a folder of timestamped
images. It does NOT detect whiteboards, does NOT run OCR - that comes
in later files (frame_detector.py, ocr_processor.py, etc).
"""

import os          # for creating folders and building file paths
import cv2         # OpenCV - the library that actually reads video files


def format_timestamp(seconds: float) -> str:
    """
    Convert a number of seconds (e.g. 42.0) into a clean "MM_SS" string
    (e.g. "00_42") that we can safely use inside a filename.

    We use underscores instead of colons because colons (:) are not
    allowed in filenames on Windows - and some of your teammates will
    likely be on Windows.

    Example:
        format_timestamp(42.0)   -> "00_42"
        format_timestamp(125.0)  -> "02_05"
    """
    total_seconds = int(seconds)          # drop any fractional part
    minutes = total_seconds // 60         # whole minutes
    remaining_seconds = total_seconds % 60  # leftover seconds after the minutes

    # :02d means "pad with a leading zero if it's a single digit"
    # so 5 seconds becomes "05", not "5"
    return f"{minutes:02d}_{remaining_seconds:02d}"


def extract_frames(video_path: str, interval_seconds: float = 2.0, output_dir: str = "output/frames"):
    """
    Open a video file and save one frame every `interval_seconds` seconds.

    Parameters
    ----------
    video_path : str
        Path to the meeting video file, e.g. "input/meeting_001.mp4"
    interval_seconds : float
        How often (in seconds) to grab a frame. Default is every 2 seconds.
        Smaller number = more frames captured = slower but more thorough.
        Larger number = fewer frames = faster but might miss something brief.
        This is deliberately a parameter (not a hardcoded value) so it's
        easy to tune later without touching the rest of the code.
    output_dir : str
        Folder where the extracted frame images will be saved.

    Returns
    -------
    list of dict
        A list like:
        [
            {"timestamp_seconds": 0.0,  "timestamp_str": "00_00", "frame_path": "output/frames/frame_00_00.jpg"},
            {"timestamp_seconds": 2.0,  "timestamp_str": "00_02", "frame_path": "output/frames/frame_00_02.jpg"},
            ...
        ]
        Every later pipeline step (detection, OCR, evidence saving) will
        work from this list instead of touching the video file again.
    """

    # ------------------------------------------------------------------
    # STEP 1: Make sure the output folder exists.
    # If it doesn't exist yet, create it (and any missing parent folders).
    # exist_ok=True means "don't error out if the folder is already there".
    # ------------------------------------------------------------------
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 2: Open the video file using OpenCV.
    # cv2.VideoCapture is OpenCV's tool for reading video files frame by frame.
    # ------------------------------------------------------------------
    video = cv2.VideoCapture(video_path)

    # If the video didn't open properly (wrong path, corrupted file, unsupported
    # codec, etc.), isOpened() will be False. We fail loudly here rather than
    # silently returning an empty list, because a silent failure is much
    # harder to debug later.
    if not video.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    # ------------------------------------------------------------------
    # STEP 3: Get basic info about the video - fps (frames per second)
    # and total frame count - so we can calculate duration and figure out
    # which exact frame numbers correspond to our desired timestamps.
    # ------------------------------------------------------------------
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard against a broken/unreadable video reporting fps as 0,
    # which would cause a division-by-zero error later.
    if fps <= 0:
        video.release()
        raise ValueError(f"Video reported an invalid fps ({fps}). File may be corrupted: {video_path}")

    duration_seconds = total_frame_count / fps

    print(f"[video_processor] Opened '{video_path}'")
    print(f"[video_processor] fps={fps:.2f}, total_frames={total_frame_count}, duration={duration_seconds:.1f}s")

    # ------------------------------------------------------------------
    # STEP 4: Walk through the video in steps of `interval_seconds`,
    # jump to that exact point in time, grab the frame, and save it.
    #
    # Instead of reading every single frame and only keeping some of them
    # (slow), we directly SEEK to the frame number we want using
    # CAP_PROP_POS_FRAMES. This is much faster for long videos.
    # ------------------------------------------------------------------
    extracted_frames = []          # this is what we'll return at the end
    current_time = 0.0             # start at the beginning of the video

    try:
        while current_time <= duration_seconds:

        # Work out which frame number corresponds to this point in time.
        # Example: at 4 seconds into a 30fps video, that's frame number 120.
            frame_number = int(current_time * fps)

        # If our calculated frame number has reached (or passed) the very
        # last real frame in the video, we're done - stop here instead of
        # trying to read a frame that doesn't exist. This is a normal,
        # expected situation at the end of every video (not an error), so
        # we quietly break instead of printing a scary warning.
            if frame_number >= total_frame_count:
                break

        # Jump the video reader directly to that frame number.
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Actually read the frame's image data.
        # success = True/False depending on whether the read worked.
        # frame_image = the actual pixel data (a NumPy array), or None if it failed.
            success, frame_image = video.read()

            if success and frame_image is not None:
            # Build a filename like "frame_00_42.jpg" using our helper function.
                timestamp_str = format_timestamp(current_time)
                frame_filename = f"frame_{timestamp_str}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

            # Save the image to disk as a .jpg file.
                if not cv2.imwrite(frame_path, frame_image):
                    raise IOError("Could not write extracted video frame")

            # Record what we just did, so later pipeline steps know
            # exactly which files exist and what timestamp each one is from.
                extracted_frames.append({
                    "timestamp_seconds": round(current_time, 2),
                    "timestamp_str": timestamp_str,
                    "frame_path": frame_path,
                })
            else:
            # This can happen right at the very end of a video if our
            # calculated frame_number slightly overshoots the last real
            # frame. We just skip it rather than crashing the whole run.
                print(f"[video_processor] Warning: could not read frame at {current_time:.1f}s - skipping.")

        # Move forward to the next sample point.
            current_time += interval_seconds
    finally:
        video.release()

    # ------------------------------------------------------------------
    # STEP 5: Clean up - release the video file so it's not left locked
    # or held open in memory.
    # ------------------------------------------------------------------
    if total_frame_count > 0 and not extracted_frames:
        raise IOError("Video opened but no frames could be decoded")

    print(f"[video_processor] Extracted {len(extracted_frames)} frames to '{output_dir}'")

    return extracted_frames


# ----------------------------------------------------------------------
# This block only runs if you execute this file directly
# (e.g. "python video_processor.py"), NOT when another file imports it
# with "from video_processor import extract_frames".
#
# It's a simple manual test: point it at a video and see what happens.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_video_path = "input/meeting_001.mp4"  # change this to your own test video
    frames = extract_frames(test_video_path, interval_seconds=2.0)

    print("\nFirst few extracted frames:")
    for frame_info in frames[:5]:
        print(frame_info)
