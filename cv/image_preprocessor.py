"""
image_preprocessor.py

PURPOSE
-------
This is Step 4 of the CV pipeline. frame_detector.py tells us WHERE a
whiteboard/slide is in a frame (4 corner points), but that region is
usually photographed at an angle - it looks like a skewed quadrilateral,
not a neat rectangle. Text on a skewed image is harder for OCR to read
accurately.

This module:
    1. Takes the original frame + the 4 detected corner points
    2. "Un-skews" that region into a flat, straight-on rectangle
       (this is called PERSPECTIVE CORRECTION)
    3. Applies a few image enhancement steps aimed at making text
       easier for OCR to read (not aimed at making the image look nice)

WHY PERSPECTIVE CORRECTION MATTERS
------------------------------------
Imagine photographing a whiteboard from the side - the far edge looks
smaller than the near edge, and the text near the far edge is
compressed and harder to read. Perspective correction mathematically
"stretches" the image back into a proper rectangle, as if the camera
had been positioned directly in front of the board. This alone can
significantly improve OCR accuracy on angled photos.
"""

import cv2
import numpy as np


def order_points(points):
    """
    Take 4 corner points in ANY order and sort them into a consistent
    order: top-left, top-right, bottom-right, bottom-left.

    WHY THIS IS NEEDED: frame_detector.py finds 4 corners, but doesn't
    guarantee which one is "first" - it could start from any corner
    and go clockwise or counter-clockwise depending on how the shape
    was drawn. To correctly "unwarp" the image, we need to know
    exactly which point is top-left, which is top-right, etc.

    THE TRICK: for a roughly rectangular shape,
        - the top-left point has the SMALLEST (x + y) sum
        - the bottom-right point has the LARGEST (x + y) sum
        - the top-right point has the SMALLEST (y - x) difference
        - the bottom-left point has the LARGEST (y - x) difference
    This is a standard, well-known trick for exactly this problem.

    Parameters
    ----------
    points : list of [x, y] (4 points, any order)

    Returns
    -------
    numpy array of shape (4, 2), in order:
        [top-left, top-right, bottom-right, bottom-left]
    """
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    point_sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sums)]  # top-left: smallest sum
    ordered[2] = points[np.argmax(point_sums)]  # bottom-right: largest sum

    point_diffs = np.diff(points, axis=1).flatten()
    ordered[1] = points[np.argmin(point_diffs)]  # top-right: smallest diff
    ordered[3] = points[np.argmax(point_diffs)]  # bottom-left: largest diff

    return ordered


def correct_perspective(frame_image, corners):
    """
    "Un-skew" the detected region into a flat, straight-on rectangle.

    Parameters
    ----------
    frame_image : numpy array
        The full original frame (as loaded by cv2.imread or a video frame).
    corners : list of [x, y]
        The 4 detected corner points, in ANY order (order_points()
        handles sorting them).

    Returns
    -------
    numpy array
        A new image containing just the board/slide region, warped
        into a proper rectangle.
    """
    ordered_corners = order_points(corners)
    (top_left, top_right, bottom_right, bottom_left) = ordered_corners

    # Work out how wide and tall the output rectangle should be, based
    # on the actual pixel distances between the corners. We take the
    # LARGER of the two width measurements (top edge vs bottom edge)
    # and the larger of the two height measurements (left edge vs
    # right edge), so we don't lose any detail by under-sizing.
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    output_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    output_height = int(max(height_left, height_right))

    # Guard against a degenerate (near-zero-size) detection.
    if output_width < 10 or output_height < 10:
        raise ValueError("Detected region is too small to perspective-correct.")

    # Define where each corner SHOULD end up in the output image: a
    # perfect rectangle from (0,0) to (output_width, output_height).
    destination_corners = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ], dtype="float32")

    # Compute the transformation that maps the skewed corners onto
    # that perfect rectangle, then apply it to the whole frame.
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
    warped = cv2.warpPerspective(frame_image, transform_matrix, (output_width, output_height))

    return warped


def enhance_for_ocr(image):
    """
    Apply preprocessing aimed at making TEXT easier for OCR to read -
    not at making the image look nice to a human.

    Steps used (each chosen for a specific OCR-readability reason):
        1. Grayscale - color doesn't help OCR and slows processing.
        2. Denoise - reduces camera/compression noise that can be
           mistaken for extra character strokes.
        3. Adaptive threshold - converts the image to pure black/white,
           which is what most OCR engines are tuned to read best. We
           use ADAPTIVE thresholding (not a single global threshold)
           because lighting is rarely even across a whole whiteboard -
           adaptive thresholding adjusts locally instead of picking
           one brightness cutoff for the entire image.

    IMPORTANT: we don't assume these steps always help - test the
    actual OCR output with and without this function and keep whichever
    performs better on your real footage (see ocr_processor.py's test,
    which does exactly this comparison).

    Returns
    -------
    numpy array : the processed image, ready to hand to an OCR engine.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    thresholded = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=10,
    )

    return thresholded


if __name__ == "__main__":
    # Quick manual test: detect + perspective-correct + enhance the
    # synthetic skewed whiteboard test image, and save both outputs
    # so we can visually confirm each step worked.
    from frame_detector import detect_visual_region

    image = cv2.imread("tests/fake_whiteboard_photo.jpg")
    if image is None:
        raise IOError("Could not read manual-test image")
    detection = detect_visual_region(image)

    if detection["detected"]:
        corrected = correct_perspective(image, detection["corners"])
        if not cv2.imwrite("tests/fake_whiteboard_corrected.jpg", corrected):
            raise IOError("Could not write corrected manual-test image")
        print("Saved tests/fake_whiteboard_corrected.jpg")

        enhanced = enhance_for_ocr(corrected)
        if not cv2.imwrite("tests/fake_whiteboard_enhanced.jpg", enhanced):
            raise IOError("Could not write enhanced manual-test image")
        print("Saved tests/fake_whiteboard_enhanced.jpg")
    else:
        print("No region detected - nothing to correct.")
