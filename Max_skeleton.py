
import numpy as np
import cv2


def select_max_bounding_area_skeleton(skeletons):
    """
    Selects the skeleton with the maximum bounding box area.

    Parameters
    ----------
    skeletons : list of np.ndarray
        List of skeletons detected in the image.
        Each skeleton is a NumPy array of shape (25, 3): [x, y, confidence].

        Example:
            [
                [[x1, y1, c1], [x2, y2, c2], ..., [x25, y25, c25]],  # Person 1
                [[x1, y1, c1], [x2, y2, c2], ..., [x25, y25, c25]],  # Person 2
                ...
            ]

    Returns
    -------
    selected_skeleton : np.ndarray
        Skeleton (25, 3) corresponding to the person with the largest
        bounding box area. Each keypoint is (x, y, confidence).

    Raises
    ------
    ValueError:
        If no skeletons are found or all detected skeletons have no
        confident keypoints (confidence <= 0.1).
    """

    # Ensure that the skeleton list is not empty
    if not skeletons:
        raise ValueError("No skeletons found to select from!")

    max_area = -1
    selected_skeleton = None

    # Iterate through all detected skeletons
    for i, skel in enumerate(skeletons):
        # Filter out low-confidence keypoints (confidence threshold = 0.1)
        valid_points = skel[skel[:, 2] > 0.1]

        # Skip this skeleton if no valid keypoints are detected
        if len(valid_points) == 0:
            continue

        # Compute the bounding box around the confident keypoints
        x_min, y_min = np.min(valid_points[:, 0]), np.min(valid_points[:, 1])
        x_max, y_max = np.max(valid_points[:, 0]), np.max(valid_points[:, 1])

        # Calculate the bounding box area
        area = (x_max - x_min) * (y_max - y_min)

        # Update if this skeleton has the largest area so far
        if area > max_area:
            max_area = area
            selected_skeleton = skel

    # Check if any skeleton was selected
    if selected_skeleton is None:
        raise ValueError("No valid skeleton with confident keypoints found.")

    return selected_skeleton