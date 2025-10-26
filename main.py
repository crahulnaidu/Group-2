import argparse

from task_11_load_image import load_image_rgb 
from task_12_skeleton_extractor import PoseEstimator
from task_131_max_skeleton import select_max_bounding_area_skeleton
from task_132_joint_angles import convert_to_joint_angles 


def main(image_path, openpose_path=None, visualize=True, output_path="output.jpg"):

    # Task 1.1: Image Acquisition and Preprocessing
    print("\n---------------Task 1.1: Image Acquisition---------------")
    image = load_image_rgb(image_path)
    
    # Task 1.2: Multi-Person 25-Keypoint Extraction
    print("\n---------------Task 1.2: Pose Extraction---------------")
    pose_estimator = PoseEstimator(openpose_path)
    skeletons = pose_estimator.extract_keys_25(image)
    
    if len(skeletons) == 0:
        raise ValueError("No skeletons detected in the image")
    
    # Task 1.3: Target Selection and Kinematic Conversion
    print("\n---------------Task 1.3: Target Selection and Conversion---------------")
    
    selected_skeleton = select_max_bounding_area_skeleton(skeletons)
    
    if selected_skeleton is None:
        raise ValueError("Couldn't select a target person")
    
    
    if visualize:
        pose_estimator.visualize_skeletons(image, selected_skeleton, skeletons, True, output_path)
    
    # Convert to joint angles
    initial_pose_vector = convert_to_joint_angles(selected_skeleton)
    
    print(f"\nInitial pose vector (θ_init) shape: {initial_pose_vector.shape}")
    print(f"Joint angles:\n{initial_pose_vector}")
    
    return initial_pose_vector


# Taking command-line arguments for input image path
parser = argparse.ArgumentParser(description="Module 1: Pose Estimation & Initial Pose Extraction")
parser.add_argument("image_path", type=str, help="Input image path") 
args = parser.parse_args()

OUTPUT_PATH = f"{args.image_path.split('.')[0]}_pose_output.jpg"
OPENPOSE_PATH = r"C:\Users\ATRI SUKUL\Desktop\IIT Tirupati\ISL\Project\openpose"

initial_pose_vector = main(image_path=args.image_path,openpose_path=OPENPOSE_PATH,visualize=True,output_path=OUTPUT_PATH)

print()
print(f"\nOutput saved to: {OUTPUT_PATH}")
