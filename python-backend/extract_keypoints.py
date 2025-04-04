

import argparse
import json
import cv2
import mediapipe as mp

def extract_keypoints_from_video(video_path, output_json):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        return

    keypoints_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_landmarks = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks.append({
                    "id": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

            keypoints_data.append({
                "frame": frame_idx,
                "landmarks": frame_landmarks
            })

        frame_idx += 1

    cap.release()
    pose.close()

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(keypoints_data, f, indent=2)

    print(f"Keypoints saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from video using MediaPipe.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="keypoints.json", help="Output JSON file path.")
    args = parser.parse_args()

    extract_keypoints_from_video(args.video, args.output)


if __name__ == "__main__":
    main()



    # import bpy
    # import sys
    # import json
    # import argparse
    #
    #
    # def create_armature_in_blender(armature_name="Armature", bone_count=33):
    #
    #     # Add a new armature object
    #     bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
    #     armature_obj = bpy.context.active_object
    #     armature_obj.name = armature_name
    #
    #     # Remove the default bone created by Blender
    #     arm = armature_obj.data
    #     default_bone = arm.edit_bones[0]
    #     arm.edit_bones.remove(default_bone)
    #
    #     # Create a simple chain of bones
    #     for i in range(bone_count):
    #         bone = arm.edit_bones.new(f"Bone_{i}")
    #         # For demonstration, place them in a vertical line
    #         bone.head = (0, i * 0.2, 0)
    #         bone.tail = (0, i * 0.2 + 0.2, 0)
    #
    #     # Switch back to Object Mode
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #     return armature_obj
    #
    #
    # def animate_armature_with_keypoints(armature_obj, keypoints_json, frame_scale=1):
    #
    #     # Load keypoints
    #     with open(keypoints_json, "r") as f:
    #         keypoints_data = json.load(f)
    #
    #     # Go to Pose Mode
    #     bpy.context.view_layer.objects.active = armature_obj
    #     bpy.ops.object.mode_set(mode='POSE')
    #     arm = armature_obj.data
    #
    #     # For each frame, set bone positions and insert keyframes
    #     for frame_info in keypoints_data:
    #         frame_idx = frame_info["frame"]
    #         landmarks = frame_info["landmarks"]
    #
    #         for lm in landmarks:
    #             bone_name = f"Bone_{lm['id']}"
    #             if bone_name not in arm.bones:
    #                 continue  # skip if mismatch
    #
    #             pose_bone = armature_obj.pose.bones.get(bone_name)
    #             if not pose_bone:
    #                 continue
    #
    #             # Convert normalized [0..1] to something in Blender space
    #             x = lm['x'] - 0.5  # shift to center
    #             y = lm['y'] - 0.5
    #             z = lm['z'] if 'z' in lm else 0.0
    #
    #             scale_factor = 2.0
    #             x *= scale_factor
    #             y *= -scale_factor  # invert Y
    #             z *= scale_factor
    #
    #             # Set the bone location in pose space
    #             pose_bone.location = (x, y, z)
    #
    #         # Insert a keyframe for each bone at this frame
    #         current_frame = frame_idx * frame_scale
    #         bpy.context.scene.frame_set(current_frame)
    #         for pose_bone in armature_obj.pose.bones:
    #             pose_bone.keyframe_insert(data_path="location", index=-1)
    #
    #     # Return to Object Mode
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #     print("Animation complete!")
    #
    #
    # def main():
    #     # Parse arguments from Blender's sys.argv
    #     argv = sys.argv
    #     if "--" in argv:
    #         # Everything after '--' are script arguments
    #         argv = argv[argv.index("--") + 1:]
    #     else:
    #         argv = []  # No script arguments
    #
    #     parser = argparse.ArgumentParser(description="Animate a Blender armature from keypoints JSON.")
    #     parser.add_argument("--keypoints", type=str, required=True, help="Path to keypoints JSON file.")
    #     args = parser.parse_args(argv)
    #
    #     # Create armature (33 bones = MediaPipe Pose)
    #     armature_obj = create_armature_in_blender(armature_name="PoseArmature", bone_count=33)
    #
    #     # Animate armature
    #     animate_armature_with_keypoints(armature_obj, args.keypoints, frame_scale=1)
    #
    #
    # if __name__ == "__main__":
    #     main()

