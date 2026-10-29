"""Shared original bounding-box and number styling for planner images."""


def draw_numbered_box(image_bgr, box, label, text_positions, *, keep_label_visible=False):
    """Draw an existing box and its label; never infer or modify a box."""
    import cv2
    height, width = image_bgr.shape[:2]
    color = (0, 0, 255)
    cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
    position = (min(box[0] + 2, max(0, width - 20)), max(12, box[1] - 4))
    for previous in text_positions:
        if abs(previous[0] - position[0]) < 10 and abs(previous[1] - position[1]) < 10:
            position = (int(box[0]) + 10, int(box[1]) - 10)
    if keep_label_visible:
        # The original overlap fallback can place text above the image.
        # Keep Skill labels in frame and separated without inventing a box.
        candidates = [position] + [(int(box[0]) + dx, int(box[1]) + dy)
                                   for dy in (12, 24, 36, 48) for dx in (2, 14, 26, 38)]
        candidates = [(max(0, min(x, max(0, width - 10))),
                       max(1, min(max(12, y), max(1, height - 1)))) for x, y in candidates]
        position = next((candidate for candidate in candidates if all(
            abs(previous[0] - candidate[0]) >= 10 or abs(previous[1] - candidate[1]) >= 10
            for previous in text_positions)), candidates[0])
    text_positions.append(position)
    cv2.putText(image_bgr, str(label), position, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, cv2.LINE_AA)
