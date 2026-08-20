"""
frame_selector.py

PURPOSE
-------
This is Step 3 of the CV pipeline. A meeting video might show the SAME
whiteboard for 20 seconds straight while someone talks. If we sampled
a frame every 2 seconds, that's 10 nearly-identical images of the same
board - we don't want 10 pieces of "evidence" for one thing that was
written once. We want ONE representative frame per distinct board
state, and a NEW piece of evidence only when the board actually
changes (new writing appears, someone flips to a new slide, etc).

WHICH SIMILARITY METHOD WE USE, AND WHY
------------------------------------------
We use PERCEPTUAL HASHING (specifically "average hash" / "phash" via
the `imagehash` library).

How it works, in plain terms: it shrinks an image down to a tiny
low-detail version (e.g. 8x8 pixels), then encodes the general pattern
of light/dark areas as a short fingerprint (a "hash"). Two images that
LOOK similar to a human end up with very similar fingerprints, even if
they're not byte-for-byte identical (slightly different camera angle,
minor lighting flicker, video compression noise, etc). We measure
"how similar" two hashes are using "Hamming distance" - basically,
how many bits differ between the two fingerprints. A small distance
means the images look almost the same; a large distance means they
look meaningfully different.

We chose this over alternatives because:
    - It's very fast (needed since we may compare many frames)
    - It's naturally tolerant of small changes (compression noise,
      tiny camera shake) without needing us to tune much
    - It's a single well-understood library call, not something we
      have to hand-build and validate ourselves

An alternative would be pixel-by-pixel comparison (e.g. SSIM), which
is more precise but much more sensitive to tiny camera movement -
it would flag "different" every time the camera shakes slightly, even
if nothing on the whiteboard actually changed. Perceptual hashing is a
better fit for "did the CONTENT change" rather than "did the exact
pixels change".

THIS FILE DOES NOT DETECT WHITEBOARDS.
It assumes you've already run frame_detector.py and are only handing
this module the frames that had something detected. Its only job is
deciding which of THOSE frames are worth keeping as unique evidence.
"""

import cv2
import imagehash
from PIL import Image


def _crop_to_detection(frame_path, detection):
    """
    Load a frame and crop it down to just the detected whiteboard/slide
    region (using the bounding box of its corner points).

    IMPORTANT LESSON LEARNED WHILE TESTING THIS MODULE:
    Our first version hashed the ENTIRE frame (board + background wall/
    room). That turned out to be too coarse - when we tested it on a
    video where the board's text genuinely changed partway through, the
    hash distance came back as 0 (i.e. "no change detected") every
    time. The problem: the board's text is a small fraction of the
    whole frame, so shrinking the full frame down to a tiny hash
    averaged the text change away completely.

    The fix: hash only the CROPPED board region, not the whole frame.
    That way, a text change on the board makes up a much bigger share
    of what's being hashed, so it actually shows up. We verified this
    fix on our test video before keeping it - see the note in the
    README / commit message.
    """
    image = cv2.imread(frame_path)
    if image is None:
        raise IOError("Could not read frame for similarity comparison")
    xs = [point[0] for point in detection["corners"]]
    ys = [point[1] for point in detection["corners"]]
    x_min, x_max = max(min(xs), 0), min(max(xs), image.shape[1])
    y_min, y_max = max(min(ys), 0), min(max(ys), image.shape[0])
    cropped_bgr = image[y_min:y_max, x_min:x_max]
    if cropped_bgr.size == 0:
        raise ValueError("Detected frame crop is empty")

    # PIL (used by the imagehash library) expects RGB, but OpenCV loads
    # images as BGR - convert so colors aren't scrambled.
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def compute_similarity(frame_path_a, detection_a, frame_path_b, detection_b, hash_size=16):
    """
    Compare the DETECTED BOARD REGION of two frames and return how
    different they are.

    We compare only the cropped board region (not the whole frame) and
    use a larger-than-default hash size (16x16 instead of the library's
    default 8x8) so smaller text changes are still picked up. Both of
    these choices came directly out of testing - see the note above.

    Returns
    -------
    int
        The Hamming distance between their perceptual hashes.
        0 means "identical" (or so close a human couldn't tell apart).
        On our test footage, frames with no real change scored 0, and
        a frame with genuinely new/changed text scored ~77 (out of a
        possible 256 with hash_size=16) - a very clear gap. Real
        footage will be noisier than our synthetic test video, so
        re-check this gap once you test on real recordings.
    """
    image_a = _crop_to_detection(frame_path_a, detection_a)
    image_b = _crop_to_detection(frame_path_b, detection_b)

    hash_a = imagehash.average_hash(image_a, hash_size=hash_size)
    hash_b = imagehash.average_hash(image_b, hash_size=hash_size)

    # The imagehash library overloads the "-" operator to directly give
    # you the Hamming distance between two hashes - this is not regular
    # subtraction, it's a built-in feature of the library.
    return hash_a - hash_b


def filter_unique_frames(detected_frames, similarity_threshold=20):
    """
    Given a list of frames that already have a detected whiteboard/slide
    region (the output of frame_detector.detect_frames_with_visual_content),
    keep only ONE representative frame per distinct board/slide state.

    How it decides: it walks through the frames in time order, and
    compares each new frame to the LAST ONE IT KEPT (not every previous
    frame - just the most recent kept one, since that's the current
    "known state" of the board). If the new frame looks basically the
    same as that last kept frame, it's a duplicate - skip it. If it
    looks meaningfully different, the board must have changed - keep it
    as new evidence.

    Parameters
    ----------
    detected_frames : list of dict
        Frames with a "detection" key already attached (output of
        frame_detector.detect_frames_with_visual_content()). Must
        already be in time order (earliest first) - extract_frames()
        naturally returns them this way.
    similarity_threshold : int
        Hamming distance (out of a max of hash_size*hash_size, 256 by
        default) above which two frames are considered DIFFERENT
        enough to count as a real change. Lower = more sensitive
        (keeps more frames). Higher = less sensitive (keeps fewer
        frames). Default 20 is based on our test video, where
        no-change frames scored 0 and a genuine change scored ~77 -
        20 sits safely in between. Re-check this once you test on
        real, noisier footage.

    Returns
    -------
    list of dict
        A subset of the input list - only the frames judged to be
        unique/representative.
    """
    if not detected_frames:
        return []

    unique_frames = [detected_frames[0]]  # always keep the first one

    for current_frame in detected_frames[1:]:
        last_kept_frame = unique_frames[-1]

        distance = compute_similarity(
            last_kept_frame["frame_path"], last_kept_frame["detection"],
            current_frame["frame_path"], current_frame["detection"],
        )

        if distance >= similarity_threshold:
            # Different enough from the last kept frame - the board
            # likely changed. Keep this one as new evidence.
            unique_frames.append(current_frame)
        # else: too similar to the last kept frame - skip it, it's a duplicate.

    print(f"[frame_selector] Kept {len(unique_frames)} unique frame(s) out of {len(detected_frames)} detected frame(s).")
    return unique_frames


if __name__ == "__main__":
    # Quick manual test: run the full chain so far (extract -> detect ->
    # filter). Uses meeting_002.mp4, our synthetic test video where the
    # board content is constant for 5s then changes once - a good
    # sanity check that duplicates get collapsed AND real changes don't.
    from video_processor import extract_frames
    from frame_detector import detect_frames_with_visual_content

    all_frames = extract_frames("input/meeting_002.mp4", interval_seconds=1.0)
    detected = detect_frames_with_visual_content(all_frames)
    unique = filter_unique_frames(detected)

    for f in unique:
        print(f["frame_path"], "->", f["detection"])
