"""
frame_detector.py

PURPOSE
-------
This is Step 2 of the CV pipeline: given a single frame (one of the
images extract_frames() saved), figure out whether it contains a
whiteboard or presentation slide, and if so, WHERE exactly it is in
the image (its 4 corner points).

WHY CLASSICAL OPENCV INSTEAD OF A TRAINED MODEL
-------------------------------------------------
A whiteboard or projected slide is usually:
    - a large, roughly rectangular region
    - noticeably brighter / higher-contrast than its surroundings
    - taking up a meaningful chunk of the frame (not a tiny corner)

Those are exactly the kind of shapes classical edge + contour
detection is good at finding, with ZERO labeled training data needed.
This is why our plan says "don't train a model yet" - we try this
first and only move to a trained detector if this proves unreliable
on our real test videos (Step 9 in the plan).

HOW THE DETECTION WORKS (high level)
-------------------------------------
1. Convert the frame to grayscale (color isn't needed to find edges).
2. Blur it slightly (reduces noise so we don't detect tiny fake edges).
3. Run Canny edge detection (highlights the outlines of shapes).
4. Thicken (dilate) those edges slightly so broken/gappy outlines
   become one continuous, closed shape.
5. Find all the closed contours (outlines) in that edge image.
6. For each contour, try to approximate it as a simple polygon.
   - If it approximates to a 4-sided shape (a quadrilateral) AND
     it's big enough to plausibly be a whiteboard/slide (not a tiny
     sticky note or a window frame) -> that's our candidate.
   - If NO clean 4-sided shape is found, we fall back to just taking
     the bounding rectangle of the largest big-enough contour. This
     is less precise but far more robust against real-world messy
     edges (shadows, reflections, marker pens sticking out, etc).
7. Pick the single largest valid candidate as "the" detected region
   for this frame (we assume one whiteboard/slide per frame for now).

THIS FILE DOES NOT DO OCR OR CROPPING/PERSPECTIVE CORRECTION.
Those are separate later steps (image_preprocessor.py, ocr_processor.py).
This file's only job is: "is there something here, and where."
"""

import cv2
import numpy as np


def detect_visual_region(frame_image, min_area_ratio=0.15):
    """
    Look for a whiteboard/slide-shaped region in a single frame image.

    Parameters
    ----------
    frame_image : numpy array
        The image to search, as loaded by cv2.imread() or returned
        from a video frame read. Must be a color (BGR) image.
    min_area_ratio : float
        The candidate region must cover at least this fraction of the
        total frame area to count (default 0.15 = at least 15% of the
        frame). This filters out small irrelevant rectangles like a
        laptop screen in the background or a picture frame on a wall.
        Tune this value once you test on real footage.

    Returns
    -------
    dict
        If something was found:
            {
                "detected": True,
                "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 4 points, not necessarily in perfect order yet
                "area_ratio": 0.42,      # how much of the frame this region covers
                "method": "polygon" or "bounding_box",  # which detection method succeeded
            }
        If nothing was found:
            {"detected": False}
    """

    frame_height, frame_width = frame_image.shape[:2]
    frame_area = frame_height * frame_width
    min_area = frame_area * min_area_ratio

    # ------------------------------------------------------------------
    # STEP 1-4: Standard OpenCV edge-detection preprocessing.
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate = "thicken" the white edge lines. Real-world edges are
    # often broken up by shadows or glare; this helps join them back
    # into one closed outline that findContours can work with.
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    # ------------------------------------------------------------------
    # STEP 5: Find all contours (closed outlines) in the edge image.
    # RETR_EXTERNAL = only outermost contours (we don't care about
    # shapes nested inside other shapes here).
    # ------------------------------------------------------------------
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"detected": False}

    # Look at the biggest contours first - a whiteboard/slide should
    # be one of the largest shapes in the frame, not a tiny detail.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_polygon_candidate = None
    best_bounding_box_candidate = None

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip anything too small to plausibly be a whiteboard/slide.
        if area < min_area:
            continue

        # ------------------------------------------------------------
        # STEP 6a: Try to simplify this contour's outline down to a
        # small number of straight edges. epsilon controls how "loose"
        # the approximation is - 2% of the contour's perimeter is a
        # commonly used, reasonable starting value.
        # ------------------------------------------------------------
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            # This is a clean 4-cornered shape - ideal case.
            if best_polygon_candidate is None:
                corners = approx.reshape(4, 2).tolist()
                best_polygon_candidate = {
                    "detected": True,
                    "corners": corners,
                    "area_ratio": round(area / frame_area, 3),
                    "method": "polygon",
                }

        # ------------------------------------------------------------
        # STEP 6b: Fallback - regardless of the polygon check above,
        # also remember the bounding box of the largest big-enough
        # contour, in case no clean 4-point polygon is ever found.
        # ------------------------------------------------------------
        if best_bounding_box_candidate is None:
            x, y, w, h = cv2.boundingRect(contour)
            corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            best_bounding_box_candidate = {
                "detected": True,
                "corners": corners,
                "area_ratio": round(area / frame_area, 3),
                "method": "bounding_box",
            }

    # Prefer the clean polygon match if we found one; otherwise fall
    # back to the bounding box of the largest big-enough contour.
    if best_polygon_candidate is not None:
        return best_polygon_candidate
    elif best_bounding_box_candidate is not None:
        return best_bounding_box_candidate
    else:
        return {"detected": False}


def classify_region_type(cropped_region_image):
    """
    VERY ROUGH first-pass heuristic to guess whether a detected region
    is a "whiteboard" or a "slide" (projected/screen presentation).

    This is intentionally simple and WILL be wrong sometimes - it's a
    starting point to refine once we run Step 9 (accuracy evaluation)
    on real test footage, not a final answer.

    Heuristic used:
        - Whiteboards are usually near-uniformly bright/white with
          low color saturation (marker text is a small % of the area).
        - Slides/screens more often have some color content (logos,
          colored text/backgrounds, photos) and sometimes a bluish
          or warm tint from the projector/screen light.

    Returns
    -------
    str : "whiteboard", "slide", or "unknown"
    """
    if cropped_region_image is None or cropped_region_image.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(cropped_region_image, cv2.COLOR_BGR2HSV)
    average_saturation = hsv[:, :, 1].mean()   # 0 = grayscale, 255 = very colorful
    average_brightness = hsv[:, :, 2].mean()   # 0 = black, 255 = white

    if average_brightness > 150 and average_saturation < 40:
        return "whiteboard"
    elif average_saturation >= 40:
        return "slide"
    else:
        return "unknown"


def detect_frames_with_visual_content(frame_list, min_area_ratio=0.15):
    """
    Run detect_visual_region() across a list of frames (the output of
    extract_frames() from video_processor.py), and return only the
    frames where something was actually detected - each annotated
    with its detection info and a first-pass type guess.

    Parameters
    ----------
    frame_list : list of dict
        Output from video_processor.extract_frames(), i.e. a list of
        {"timestamp_seconds", "timestamp_str", "frame_path"} dicts.
    min_area_ratio : float
        Passed through to detect_visual_region().

    Returns
    -------
    list of dict
        Same dicts as the input, but only for frames where something
        was detected, with an extra "detection" key added, e.g.:
        {
            "timestamp_seconds": 4.0,
            "timestamp_str": "00_04",
            "frame_path": "output/frames/frame_00_04.jpg",
            "detection": {
                "detected": True,
                "corners": [...],
                "area_ratio": 0.42,
                "method": "polygon",
                "type": "whiteboard",
            }
        }
    """
    results = []

    for frame_info in frame_list:
        image = cv2.imread(frame_info["frame_path"])

        if image is None:
            raise IOError("Could not read an extracted frame for detection")

        detection = detect_visual_region(image, min_area_ratio=min_area_ratio)

        if detection["detected"]:
            # Crop out just the detected region (using its bounding
            # box) so classify_region_type() only looks at that area,
            # not the whole frame.
            xs = [point[0] for point in detection["corners"]]
            ys = [point[1] for point in detection["corners"]]
            x_min, x_max = max(min(xs), 0), min(max(xs), image.shape[1])
            y_min, y_max = max(min(ys), 0), min(max(ys), image.shape[0])
            cropped = image[y_min:y_max, x_min:x_max]

            detection["type"] = classify_region_type(cropped)

            frame_with_detection = dict(frame_info)   # copy, don't mutate the original
            frame_with_detection["detection"] = detection
            results.append(frame_with_detection)

    print(f"[frame_detector] {len(results)} of {len(frame_list)} frames had a detected visual region.")
    return results


if __name__ == "__main__":
    # Quick manual test: run detection on whatever frames are already
    # sitting in output/frames/ (created by running video_processor.py).
    import glob

    frame_paths = sorted(glob.glob("output/frames/frame_*.jpg"))
    fake_frame_list = [{"timestamp_seconds": 0, "timestamp_str": "manual", "frame_path": p} for p in frame_paths]

    detected = detect_frames_with_visual_content(fake_frame_list)
    for item in detected:
        print(item["frame_path"], "->", item["detection"])
