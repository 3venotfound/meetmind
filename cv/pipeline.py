"""
pipeline.py

PURPOSE
-------
This is Step 7 of the CV pipeline - and the FINAL piece Maryam's CV
module needs. It wires together every module built so far into one
function:

    process_meeting_video("meeting_001.mp4", "M001")

which returns exactly the JSON structure the backend/AI team expects:

    {
        "meeting_id": "M001",
        "visual_evidence": [
            {
                "timestamp": "00:42",
                "type": "whiteboard",
                "text": "Launch Date: 15 August\\nBudget: RM70,000",
                "confidence": 0.91,
                "image_path": "evidence/M001_00_42_whiteboard.jpg"
            },
            ...
        ]
    }

THE FULL CHAIN, STEP BY STEP:
    1. video_processor.extract_frames()
       -> sample frames from the video every N seconds
    2. frame_detector.detect_frames_with_visual_content()
       -> keep only frames with a whiteboard/slide detected
    3. frame_selector.filter_unique_frames()
       -> collapse near-duplicate frames down to one per board state
    4. For each unique frame:
       a. image_preprocessor.correct_perspective()
          -> un-skew the detected region
       b. ocr_processor.run_ocr()
          -> extract the text (on the RAW corrected crop - see the
             note in ocr_processor.py on why we skip enhancement by
             default)
       c. evidence_manager.save_evidence()
          -> save the image + build the JSON record

THIS FILE DOES NOT CONNECT TO A DATABASE OR API.
Per the plan: "The first version must work independently." Whoever
calls process_meeting_video() (you, in a test script, or later a
FastAPI endpoint) is responsible for what happens to the result.
"""

from video_processor import extract_frames
from frame_detector import detect_frames_with_visual_content, classify_region_type
from frame_selector import filter_unique_frames
from image_preprocessor import correct_perspective
from ocr_processor import run_ocr
from evidence_manager import save_evidence


def process_meeting_video(video_path, meeting_id, interval_seconds=2.0):
    """
    Run the full CV pipeline on a meeting video and return structured
    visual evidence.

    Parameters
    ----------
    video_path : str
        Path to the meeting video, e.g. "input/meeting_001.mp4".
    meeting_id : str
        An identifier for this meeting, e.g. "M001". Used to name
        saved evidence files.
    interval_seconds : float
        How often to sample frames from the video. Passed straight
        through to video_processor.extract_frames() - see that file
        for the tradeoffs of changing it.

    Returns
    -------
    dict : {"meeting_id": ..., "visual_evidence": [...]}
    """
    print(f"[pipeline] Processing '{video_path}' as meeting '{meeting_id}'...")

    # Step 1: sample frames from the video.
    all_frames = extract_frames(video_path, interval_seconds=interval_seconds)

    # Step 2: keep only frames where a whiteboard/slide was detected.
    detected_frames = detect_frames_with_visual_content(all_frames)

    # Step 3: collapse near-duplicates down to one frame per board state.
    unique_frames = filter_unique_frames(detected_frames)

    # Step 4: for each unique frame, correct + OCR + save evidence.
    visual_evidence = []

    for frame_info in unique_frames:
        import cv2  # local import keeps this file's top readable; cv2 is only needed here
        original_image = cv2.imread(frame_info["frame_path"])
        detection = frame_info["detection"]

        try:
            corrected = correct_perspective(original_image, detection["corners"])
        except ValueError as error:
            # correct_perspective raises this if the detected region
            # was too small/degenerate to warp. Skip this frame rather
            # than crashing the whole pipeline over one bad detection.
            print(f"[pipeline] Skipping {frame_info['frame_path']}: {error}")
            continue

        # Re-classify type on the CORRECTED image rather than reusing
        # the guess from detection - the straightened, cropped image
        # is a cleaner input for this heuristic than the original
        # skewed frame was.
        region_type = classify_region_type(corrected)

        ocr_result = run_ocr(corrected)

        evidence = save_evidence(
            meeting_id=meeting_id,
            timestamp_str=frame_info["timestamp_str"],
            region_type=region_type,
            corrected_image=corrected,
            ocr_result=ocr_result,
        )
        visual_evidence.append(evidence)

    print(f"[pipeline] Done. {len(visual_evidence)} piece(s) of visual evidence produced.")

    return {
        "meeting_id": meeting_id,
        "visual_evidence": visual_evidence,
    }


if __name__ == "__main__":
    import json

    result = process_meeting_video("input/meeting_003.mp4", "M_TEST_PIPELINE")
    print("\nFinal structured output:")
    print(json.dumps(result, indent=2))
