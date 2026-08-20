"""
ocr_processor.py

PURPOSE
-------
This is Step 5 of the CV pipeline: take a perspective-corrected image
of a whiteboard/slide and extract the actual TEXT written on it.

WHICH OCR LIBRARY WE USE, AND WHY
------------------------------------
The plan called for testing PaddleOCR first (best accuracy on mixed
print/handwriting), with EasyOCR as a documented fallback in case of
install trouble.

WHAT ACTUALLY HAPPENED WHEN WE TESTED BOTH:
    - PaddleOCR: the Python package installs fine, but the first time
      you run it, it needs to DOWNLOAD its model weights from the
      internet. In this development/testing environment, that download
      failed ("No available model hosting platforms detected").
    - EasyOCR: installed cleanly and its model download succeeded
      without any issue.

IMPORTANT CAVEAT: this doesn't necessarily mean PaddleOCR won't work
on YOUR laptop - this sandboxed testing environment has a restricted
network allowlist that's stricter than a normal home/campus internet
connection, so PaddleOCR's failure here could be specific to this
sandbox rather than the library itself. Worth a quick try on your own
machine (see the commented-out PaddleOCR section below) - if it works
there, you'd get better handwriting accuracy. But since EasyOCR is
CONFIRMED working, we're using it as the default so the pipeline
isn't blocked on an install that might not go smoothly during crunch
time before the 22nd. This is exactly the "reliability > complexity"
principle from the plan - use what's proven to work, upgrade later
only if there's time.

WHAT THIS FILE DOES NOT DO
----------------------------
It doesn't decide GOOD text from BAD text, or try to fix OCR mistakes.
It just runs the OCR engine and reports back what it found, with a
confidence score - evidence_manager.py (the next module) decides what
to do with that.
"""

import easyocr

# Loading the OCR model is slow (a few seconds), so we do it ONCE at
# import time and reuse the same `_reader` object for every image,
# instead of reloading the model every time run_ocr() is called.
_reader = easyocr.Reader(['en'], gpu=False, verbose=False)


def run_ocr(image):
    """
    Run OCR on a single image and return the text found, along with
    per-line confidence scores.

    Parameters
    ----------
    image : numpy array
        The image to read text from. Works on both the raw cropped
        region and the enhanced (thresholded) version from
        image_preprocessor.enhance_for_ocr() - compare both on your
        real footage to see which gives better results (see the test
        at the bottom of this file for exactly that comparison).

    Returns
    -------
    dict
        {
            "text": "Launch Date: 15 August\nBudget: RM70,000",  # all detected lines joined together
            "lines": [
                {"text": "Launch Date: 15 August", "confidence": 0.93},
                {"text": "Budget: RM70,000", "confidence": 0.89},
            ],
            "average_confidence": 0.91,
        }
    """
    # easyocr.Reader.readtext() returns a list of
    # (bounding_box, text, confidence) tuples, one per detected line.
    results = _reader.readtext(image)

    lines = []
    for _bounding_box, text, confidence in results:
        lines.append({"text": text, "confidence": round(float(confidence), 3)})

    combined_text = "\n".join(line["text"] for line in lines)
    average_confidence = (
        round(sum(line["confidence"] for line in lines) / len(lines), 3)
        if lines else 0.0
    )

    return {
        "text": combined_text,
        "lines": lines,
        "average_confidence": average_confidence,
    }


# ----------------------------------------------------------------------
# OPTIONAL: PaddleOCR, if you want to try it on a machine with full
# internet access (uncomment and `pip install paddlepaddle paddleocr`
# first). Kept here, unused, so switching is a one-line change later
# rather than a rewrite:
#
# from paddleocr import PaddleOCR
# _paddle_reader = PaddleOCR(use_angle_cls=False, lang='en')
#
# def run_ocr_paddle(image):
#     result = _paddle_reader.ocr(image)
#     ...
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Test: compare OCR accuracy on the RAW cropped/corrected image vs
    # the ENHANCED (thresholded) version, against known ground-truth
    # text, using Character Error Rate (CER) - exactly what the plan's
    # evaluation step calls for. Lower CER = more accurate.
    import cv2
    import jiwer
    from frame_detector import detect_visual_region
    from image_preprocessor import correct_perspective, enhance_for_ocr

    ground_truth_text = "Launch Date: 15 August\nBudget: RM70,000"

    image = cv2.imread("tests/realistic_whiteboard_photo.jpg")
    if image is None:
        raise IOError("Could not read OCR manual-test image")
    detection = detect_visual_region(image)
    corrected = correct_perspective(image, detection["corners"])
    enhanced = enhance_for_ocr(corrected)

    raw_result = run_ocr(corrected)
    enhanced_result = run_ocr(enhanced)

    raw_cer = jiwer.cer(ground_truth_text, raw_result["text"])
    enhanced_cer = jiwer.cer(ground_truth_text, enhanced_result["text"])

    print("Ground truth:\n", ground_truth_text)
    print("\n--- RAW (perspective-corrected only) ---")
    print("OCR read:\n", raw_result["text"])
    print(f"Average confidence: {raw_result['average_confidence']}")
    print(f"Character Error Rate: {raw_cer:.3f}")

    print("\n--- ENHANCED (grayscale + denoise + threshold) ---")
    print("OCR read:\n", enhanced_result["text"])
    print(f"Average confidence: {enhanced_result['average_confidence']}")
    print(f"Character Error Rate: {enhanced_cer:.3f}")
