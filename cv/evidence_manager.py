"""
evidence_manager.py

PURPOSE
-------
This is Step 6 of the CV pipeline. By this point we have, for one
unique board/slide state:
    - which frame it came from, and when
    - the detected region + a whiteboard/slide type guess
    - the perspective-corrected image
    - the OCR text extracted from it

This module's only job is to turn all of that into the exact JSON
format the backend/AI team expects (our "integration contract"), and
save the evidence image to disk where they can find it.

WE DO NOT CHANGE THE CONTRACT FORMAT HERE.
Per the plan: "If you believe the format should change, explain why
before changing it." This module produces exactly:
    {
        "timestamp": "00:42",
        "type": "whiteboard",
        "text": "Launch Date: 15 August\\nBudget: RM70,000",
        "confidence": 0.91,
        "image_path": "evidence/M001_00_42_whiteboard.jpg"
    }
"""

import os
import cv2


def _timestamp_str_to_colon_format(timestamp_str):
    """
    Convert our internal filename-safe timestamp format ("00_42") into
    the colon format the integration contract expects ("00:42").

    WHY WE HAVE TWO FORMATS: colons (:) aren't allowed in filenames on
    Windows, so video_processor.py uses underscores when naming saved
    frame files. But the contract's example JSON uses colons, since
    that's the more natural way to display a timestamp. This function
    is the one place that converts between the two, so the rest of
    the pipeline doesn't have to think about it.
    """
    return timestamp_str.replace("_", ":")


def save_evidence(meeting_id, timestamp_str, region_type, corrected_image, ocr_result, evidence_dir="evidence"):
    """
    Save one piece of visual evidence: write the image to disk and
    build its JSON record.

    Parameters
    ----------
    meeting_id : str
        e.g. "M001" - used to name the saved evidence file.
    timestamp_str : str
        Our internal "00_42" format (from video_processor.py / frame_detector.py).
    region_type : str
        "whiteboard", "slide", or "unknown" (from frame_detector.classify_region_type).
    corrected_image : numpy array
        The perspective-corrected image (from image_preprocessor.correct_perspective).
    ocr_result : dict
        The output of ocr_processor.run_ocr() - must contain "text" and "average_confidence".
    evidence_dir : str
        Folder to save evidence images into.

    Returns
    -------
    dict
        One evidence record, in the exact format the backend/AI team expects:
        {
            "timestamp": "00:42",
            "type": "whiteboard",
            "text": "...",
            "confidence": 0.91,
            "image_path": "evidence/M001_00_42_whiteboard.jpg"
        }
    """
    os.makedirs(evidence_dir, exist_ok=True)

    filename = f"{meeting_id}_{timestamp_str}_{region_type}.jpg"
    image_path = os.path.join(evidence_dir, filename)
    cv2.imwrite(image_path, corrected_image)

    return {
        "timestamp": _timestamp_str_to_colon_format(timestamp_str),
        "type": region_type,
        "text": ocr_result["text"],
        "confidence": ocr_result["average_confidence"],
        "image_path": image_path,
    }


if __name__ == "__main__":
    # Quick manual test: build one evidence record end-to-end from our
    # realistic test image and check the JSON shape is exactly right.
    import cv2
    from frame_detector import detect_visual_region, classify_region_type
    from image_preprocessor import correct_perspective
    from ocr_processor import run_ocr

    image = cv2.imread("tests/realistic_whiteboard_photo.jpg")
    detection = detect_visual_region(image)
    corrected = correct_perspective(image, detection["corners"])
    region_type = classify_region_type(corrected)
    ocr_result = run_ocr(corrected)

    evidence = save_evidence(
        meeting_id="M_TEST",
        timestamp_str="00_42",
        region_type=region_type,
        corrected_image=corrected,
        ocr_result=ocr_result,
    )

    import json
    print(json.dumps(evidence, indent=2))
