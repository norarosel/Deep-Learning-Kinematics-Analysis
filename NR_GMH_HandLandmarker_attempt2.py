# Nora Cécile Rosel Zaballos - 05/06/2025 - Now with HandLandmarker
# 13/06/2025: Box + 3D Animation
# 16/06/2025: Added kinematic analysis, smoothness, speed, finger independence, joint compensation
# 25/06/2025: Finalized kinematics, comparison with healthy patient

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from matplotlib.animation import FuncAnimation
import math

# 1. File paths
video_patient_path = "./Videos/patient_1.mp4"
video_healthy_path = "./Videos/healthy_patient_1.mp4"
output_patient_path = "./patient/annotated_output_patient.mp4"
output_healthy_path = "./healthy/annotated_output_healthy.mp4"
model_path = "./hand_landmarker.task"
output_folder_patient = "./patient"
output_folder_healthy = "./healthy"

# Initialize MediaPipe hand detection model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Box points - TO CHANGE IF THE VIDEO IS DIFFERENT, although I kept the same points, the scales should not change from video to video
pt1 = (202, 515)
pt2 = (1541, 774)
pt3 = (367, 795)
pt4 = (196, 796)
pt5 = (1247, 864)
pt6 = (1549, 865)

def process_video(video_path, output_path, model_path, pt1, pt2, pt3, pt4, pt5, pt6):

    # 2. Load the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}, Resolution: {width}x{height}")

    # 3. Set up the output video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"ERROR: Could not create output video {output_path}")

    # 4. Pixel-to-cm scales
    def pixel_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Real-world distances in cm (provided)
    real_dist_13_2 = 13.2
    real_dist_6_2 = 6.2

    # Pixel distances left side (hand 2)
    pix_dist_1_4 = pixel_dist(pt1, pt4)
    pix_dist_4_3 = pixel_dist(pt4, pt3)
    # Pixel distances right side (hand 1)
    pix_dist_5_6 = pixel_dist(pt5, pt6)
    pix_dist_6_2_pix = pixel_dist(pt6, pt2)

    # Calculate pixel-to-cm scale factors
    scale_x_left = real_dist_13_2 / pix_dist_1_4
    scale_y_left = real_dist_6_2 / pix_dist_4_3
    scale_x_right = real_dist_13_2 / pix_dist_5_6
    scale_y_right = real_dist_6_2 / pix_dist_6_2_pix

    print(f"Left side scales: x={scale_x_left:.4f} cm/px, y={scale_y_left:.4f} cm/px")
    print(f"Right side scales: x={scale_x_right:.4f} cm/px, y={scale_y_right:.4f} cm/px")

    # Approximate Z scale as average of X and Y scales on each side:
    scale_z_left = (scale_x_left + scale_y_left) / 2
    scale_z_right = (scale_x_right + scale_y_right) / 2
    print(f"Left side scale Z approx: {scale_z_left:.4f} cm/px")
    print(f"Right side scale Z approx: {scale_z_right:.4f} cm/px")

    # 5. Initialize MediaPipe hand detection model
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 6. Run detection, annonate video and save results
    all_hands_landmarks = []

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            # print(f"\nProcessing frame {frame_index}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int((frame_index / fps) * 1000)

            detection_result = landmarker.detect_for_video(mp_image, timestamp)

            if detection_result.hand_landmarks:
                frame_landmarks = []

                # Get average X position for each detected hand
                hands_with_avg_x = []
                for landmarks_list in detection_result.hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks_list]
                    avg_x = np.mean([lm[0] for lm in landmarks])
                    hands_with_avg_x.append((avg_x, landmarks))

                # Sort: right side (higher X) first, left side (lower X) second
                hands_with_avg_x.sort(reverse=True, key=lambda x: x[0])

                # Assign: hand1 = right side, hand2 = left side
                for _, landmarks in hands_with_avg_x:
                    # Assume both are LEFT hands in anatomical sense
                    frame_landmarks.append(landmarks)

                # Draw landmarks
                for idx, landmarks in enumerate(frame_landmarks):
                    converted_landmarks = [
                        landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2])
                        for lm in landmarks
                    ]
                    landmark_proto = landmark_pb2.NormalizedLandmarkList(landmark=converted_landmarks)

                    mp_drawing.draw_landmarks(
                        frame,
                        landmark_proto,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                for i, lm in enumerate(landmarks):
                    cx, cy = int(lm[0] * width), int(lm[1] * height)
                    cv2.putText(frame, f"H{idx+1}:{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            else:
                print("No hands detected.")

            all_hands_landmarks.append(frame_landmarks)
            out.write(frame)
            frame_index += 1

    cap.release()
    out.release()

    scales = {
        'scale_x_left': scale_x_left,
        'scale_y_left': scale_y_left,
        'scale_z_left': scale_z_left,
        'scale_x_right': scale_x_right,
        'scale_y_right': scale_y_right,
        'scale_z_right': scale_z_right,
        'width': width,
        'height': height,
        'fps': fps
    }

    return all_hands_landmarks, scales

# 7. Process the hand landmarks in real-world coordinates (cm)
# Helper functions to convert hand landmarks from pixel coordinates to real-world cm coordinates
def convert_hand_to_cm(hand_px, scale_x, scale_y, scale_z, ref_px):
    converted = []
    for x_px, y_px, z_px in hand_px:
        x_cm = (x_px - ref_px[0]) * scale_x
        y_cm = (y_px - ref_px[1]) * scale_y
        z_cm = z_px * (scale_z)  # z from mediapipe normalized depth scaled approx
        converted.append([x_cm, y_cm, z_cm])
    return np.array(converted)

def plot_hand_3d(ax, hand, color, label):
    xs, ys, zs = hand[:, 0], hand[:, 1], hand[:, 2]
    ax.scatter(xs, ys, zs, c=color, label=label)
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, str(i), color=color, fontsize=8)
    for start_idx, end_idx in HAND_CONNECTIONS:
        x_vals = [hand[start_idx][0], hand[end_idx][0]]
        y_vals = [hand[start_idx][1], hand[end_idx][1]]
        z_vals = [hand[start_idx][2], hand[end_idx][2]]
        ax.plot(x_vals, y_vals, z_vals, c=color)

# Extract hands frames for plotting and fusion
def compute_fused_hands(all_hands_landmarks, scales, pt_left, pt_right):
    fused_hands_all_frames_cm = []

    for frame_landmarks in all_hands_landmarks:
        if len(frame_landmarks) == 2:
            # Calculate average x pixel coordinates for each detected hand
            avg_x_0 = np.mean([lm[0] for lm in frame_landmarks[0]]) * scales['width']
            avg_x_1 = np.mean([lm[0] for lm in frame_landmarks[1]]) * scales['width']

            # Determine which hand is right and which is left based on avg_x
            if avg_x_0 > avg_x_1:
                hand1_norm = frame_landmarks[0]  # right hand
                hand2_norm = frame_landmarks[1]  # left hand
            else:
                hand1_norm = frame_landmarks[1]  # right hand
                hand2_norm = frame_landmarks[0]  # left hand

            # Convert to cm with real-world x,y from hand1 reference pt5 (from the box, right side), z scaled approx for hand1
            hand1_px = [(lm[0] * scales['width'], lm[1] * scales['height'], lm[2]) for lm in hand1_norm]
            hand2_px = [(lm[0] * scales['width'], lm[1] * scales['height'], lm[2]) for lm in hand2_norm]

            hand1_cm = convert_hand_to_cm(hand1_px, scales['scale_x_right'], scales['scale_y_right'], scales['scale_z_right'], pt_right)
            hand2_cm = convert_hand_to_cm(hand2_px, scales['scale_x_left'], scales['scale_y_left'], scales['scale_z_left'], pt_left)

            fused_cm = []
            # Fused hand by averaging mediapipe z's (index matched) and using hand1's x,y
            for i in range(len(hand1_cm)):
                x = hand1_cm[i, 0]
                y = hand1_cm[i, 1]
                z = (hand1_cm[i, 2] + hand2_cm[i, 2]) / 2
                fused_cm.append([x, y, z])

            fused_hands_all_frames_cm.append(np.array(fused_cm))
        else:
            fused_hands_all_frames_cm.append(None)

    return fused_hands_all_frames_cm

# 8. Visualize the hand landmarks in real-world coordinates (cm)

import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_plots_and_animation(
    all_hands_landmarks, fused_hands_all_frames_cm, scales,
    video_path, output_dir, pt_left, pt_right
):
    ensure_dir(output_dir)
    
    width, height = scales['width'], scales['height']
    
    # Find first valid frame index with fused data
    first_valid_idx = next((i for i, f in enumerate(fused_hands_all_frames_cm) if f is not None), None)
    if first_valid_idx is None:
        print(f"No valid fused frames for {video_path}. Skipping plots.")
        return
    
    frame_landmarks = all_hands_landmarks[first_valid_idx]
    if len(frame_landmarks) != 2:
        print(f"Frame {first_valid_idx} does not have two hands detected. Skipping plots.")
        return
    
    # Calculate avg x pixel coordinates once for hand ordering
    avg_x_0 = np.mean([lm[0] for lm in frame_landmarks[0]]) * width
    avg_x_1 = np.mean([lm[0] for lm in frame_landmarks[1]]) * width

    if avg_x_0 > avg_x_1:
        hand1_norm = frame_landmarks[0]  # right hand
        hand2_norm = frame_landmarks[1]  # left hand
    else:
        hand1_norm = frame_landmarks[1]  # right hand
        hand2_norm = frame_landmarks[0]  # left hand

    # Convert landmarks to pixel coords
    hand1_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand1_norm]
    hand2_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand2_norm]

    # Convert to real-world cm coords
    hand1_cm = convert_hand_to_cm(hand1_px, scales['scale_x_right'], scales['scale_y_right'], scales['scale_z_right'], pt_right)
    hand2_cm = convert_hand_to_cm(hand2_px, scales['scale_x_left'], scales['scale_y_left'], scales['scale_z_left'], pt_left)

    # === Plot Hand 1 Frame 0 ===
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_hand_3d(ax, hand1_cm, 'red', 'Hand 1 Frame 0')
    ax.set_title("Hand 1 Frame 0 (Real World Coordinates)")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hand1_frame0_real_world.png"))
    plt.close(fig)

    # === Plot Hand 2 Frame 0 ===
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_hand_3d(ax, hand2_cm, 'blue', 'Hand 2 Frame 0')
    ax.set_title("Hand 2 Frame 0 (Real World Coordinates)")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hand2_frame0_real_world.png"))
    plt.close(fig)

    # === Plot Fused Hand Frame 0 ===
    fused_cm = fused_hands_all_frames_cm[first_valid_idx]
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_hand_3d(ax, fused_cm, 'green', 'Fused Hand Frame 0')
    ax.set_title("Fused Hand Frame 0 (Real World Coordinates)")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.view_init(elev=40, azim=120)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fused_hand_frame0.png"))
    plt.close(fig)

    # === Prepare animation limits ===
    frames_to_use = [f for f in fused_hands_all_frames_cm if f is not None]

    # Use first valid frame for limits
    x_min, x_max = frames_to_use[0][:, 0].min(), frames_to_use[0][:, 0].max()
    y_min, y_max = frames_to_use[0][:, 1].min(), frames_to_use[0][:, 1].max()
    z_min, z_max = frames_to_use[0][:, 2].min(), frames_to_use[0][:, 2].max()

    pad = 1.0
    x_min, x_max = x_min - pad, x_max + pad
    y_min, y_max = y_min - pad, y_max + pad
    z_min, z_max = z_min - pad, z_max + pad

    fig_anim = plt.figure(figsize=(10,7))
    ax_anim = fig_anim.add_subplot(111, projection='3d')

    def init_anim():
        ax_anim.set_title("Fused Hand Animation (Real World Coordinates)")
        ax_anim.set_xlabel("X (cm)")
        ax_anim.set_ylabel("Y (cm)")
        ax_anim.set_zlabel("Z (cm)")
        ax_anim.set_xlim(x_min, x_max)
        ax_anim.set_ylim(y_min, y_max)
        ax_anim.set_zlim(z_min, z_max)
        return []

    def update_anim(frame_idx):
        ax_anim.clear()
        ax_anim.set_title(f"Fused Hand Frame {frame_idx}")
        ax_anim.set_xlabel("X (cm)")
        ax_anim.set_ylabel("Y (cm)")
        ax_anim.set_zlabel("Z (cm)")
        ax_anim.set_xlim(x_min, x_max)
        ax_anim.set_ylim(y_min, y_max)
        ax_anim.set_zlim(z_min, z_max)

        if frame_idx >= len(frames_to_use):
            return []

        hand = frames_to_use[frame_idx]
        xs, ys, zs = hand[:, 0], hand[:, 1], hand[:, 2]
        ax_anim.scatter(xs, ys, zs, c='green')

        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax_anim.text(x, y, z, str(i), color='green', fontsize=8)

        for start_idx, end_idx in HAND_CONNECTIONS:
            x_vals = [hand[start_idx][0], hand[end_idx][0]]
            y_vals = [hand[start_idx][1], hand[end_idx][1]]
            z_vals = [hand[start_idx][2], hand[end_idx][2]]
            ax_anim.plot(x_vals, y_vals, z_vals, c='green')

        return []

    anim_filename = os.path.join(output_dir, "fused_hand_animation_all_frames_cm.mp4")
    anim = FuncAnimation(fig_anim, update_anim, frames=len(frames_to_use), init_func=init_anim, blit=False)
    anim.save(anim_filename, fps=15)
    plt.close(fig_anim)
    print(f"Saved full animation video: {anim_filename}")

    # === Save 2D overlay image of fused hand on frame 0 ===
    cap0 = cv2.VideoCapture(video_path)
    success, frame0_img = cap0.read()
    cap0.release()

    if success and len(all_hands_landmarks[first_valid_idx]) == 2:
        hand1 = all_hands_landmarks[first_valid_idx][0]
        fused_color = (0, 255, 255)  # Yellow

        for i in range(len(hand1)):
            x_norm = hand1[i][0]
            y_norm = hand1[i][1]

            x_px = int(x_norm * width)
            y_px = int(y_norm * height)
            cv2.circle(frame0_img, (x_px, y_px), 4, fused_color, -1)
            cv2.putText(frame0_img, str(i), (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fused_color, 1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            x0 = int(hand1[start_idx][0] * width)
            y0 = int(hand1[start_idx][1] * height)
            x1 = int(hand1[end_idx][0] * width)
            y1 = int(hand1[end_idx][1] * height)
            cv2.line(frame0_img, (x0, y0), (x1, y1), fused_color, 1)

        overlay_path = os.path.join(output_dir, "frame0_fused_hand_2d.png")
        cv2.imwrite(overlay_path, frame0_img)
        print(f"Saved 2D overlay of fused hand on frame 0 to: {overlay_path}")
    else:
        print("Could not generate 2D fused overlay for frame 0.")

# 9. Kinematic Analysis
# 9.1: Speed = distance between frames * fps (1/t)

# Calculate instantaneous speed of each landmark per frame (cm/s)
def compute_speeds(all_frames, fps):
    speeds = []

    dt = 1 / fps  # Time between frames in seconds

    for i in range(1, len(all_frames)):
        frame_prev = all_frames[i - 1]
        frame_curr = all_frames[i]

        if frame_prev is None or frame_curr is None:
            speeds.append(None)
            continue

        frame_speeds = []
        for lm_idx in range(len(frame_curr)):
            x1, y1, z1 = frame_prev[lm_idx]
            x2, y2, z2 = frame_curr[lm_idx]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            speed = distance / dt  # cm/s
            frame_speeds.append(speed)

        speeds.append(frame_speeds)

    return speeds  # List of [ [speed_lm0, speed_lm1, ...], ... ]

# Calculate avg of all landmark speeds per frame
def average_hand_speed(speeds):
    return [np.nanmean(frame) if frame is not None else None for frame in speeds]

# Calculate avg speed of all landmarks over entire video
def average_hand_speed_over_time(avg_speed_list):
    # avg_speed_list is the output of average_hand_speed(speeds)
    vals = [v for v in avg_speed_list if v is not None]
    return np.nanmean(vals) if vals else None

# Calculate speeds of each finger
finger_indices = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20]
}

def finger_speeds(speeds):
    finger_speed_frames = []

    for frame in speeds:
        if frame is None:
            finger_speed_frames.append(None)
            continue

        finger_frame = {}
        for finger, indices in finger_indices.items():
            vals = [frame[i] for i in indices]
            finger_frame[finger] = np.nanmean(vals)
        finger_speed_frames.append(finger_frame)

    return finger_speed_frames  # List of dicts { 'Thumb': val, ... }

#Whole hand average speed
def average_hand_speed_over_time(avg_speed_list):
    # avg_speed_list is output of average_hand_speed(speeds)
    vals = [v for v in avg_speed_list if v is not None]
    return np.nanmean(vals) if vals else None

# Average speed per finger over the whole video
def average_finger_speed_over_time(finger_speed_frames):
    # finger_speed_frames is the output of finger_speeds(speeds)
    avg_finger_speeds = {}
    for finger in finger_indices.keys():
        vals = [frame[finger] for frame in finger_speed_frames if frame is not None]
        avg_finger_speeds[finger] = np.nanmean(vals) if vals else None
    return avg_finger_speeds

# Max and min speeds per finger
def max_finger_speed(finger_speed_frames):
    max_speeds = {}
    for finger in finger_indices.keys():
        vals = [frame[finger] for frame in finger_speed_frames if frame is not None]
        max_speeds[finger] = np.nanmax(vals) if vals else None
    return max_speeds

#Speed variability
def std_finger_speed(finger_speed_frames):
    std_speeds = {}
    for finger in finger_indices.keys():
        vals = [frame[finger] for frame in finger_speed_frames if frame is not None]
        std_speeds[finger] = np.nanstd(vals) if vals else None
    return std_speeds

### Plot Average Hand Speed Over Time ###
def plot_avg_hand_speed_over_time(avg_speed_patient, avg_speed_healthy, output_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(avg_speed_patient, label='Patient', color='red')
    plt.plot(avg_speed_healthy, label='Healthy', color='green')
    plt.title('Average Hand Speed Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Speed (cm/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'avg_hand_speed_over_time.png'))
    plt.close()

### Plot Finger Speeds Over Time ###
def plot_finger_speeds_over_time(finger_speed_patient, finger_speed_healthy, output_folder):
    fingers = list(finger_indices.keys())

    patient_data = {finger: [frame[finger] if frame else None for frame in finger_speed_patient] for finger in fingers}
    healthy_data = {finger: [frame[finger] if frame else None for frame in finger_speed_healthy] for finger in fingers}

    plt.figure(figsize=(12, 6))
    for finger in fingers:
        plt.plot(patient_data[finger], label=f'{finger} (Patient)', linestyle='--')
        plt.plot(healthy_data[finger], label=f'{finger} (Healthy)', linestyle='-')

    plt.title('Finger Speeds Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Speed (cm/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'finger_speeds_over_time.png'))
    plt.close()


### Bar Plot Summary (Avg, Max, Std) ###
def bar_plot_finger_summary(patient_values, healthy_values, title, ylabel, filename, output_folder):
    fingers = list(finger_indices.keys())
    x = np.arange(len(fingers))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, [patient_values[f] for f in fingers], width, label='Patient', color='red')
    plt.bar(x + width/2, [healthy_values[f] for f in fingers], width, label='Healthy', color='green')

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, fingers)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

# 9.2: Calculate the dominant axis of movement

def compute_dominant_axis(fused_hands):
    deltas = []
    for frame in fused_hands:
        if frame is not None:
            deltas.append(frame.mean(axis=0))
    deltas = np.array(deltas)
    range_per_axis = np.ptp(deltas, axis=0)
    dominant_axis = ['X', 'Y', 'Z'][np.argmax(range_per_axis)]
    return dominant_axis, range_per_axis

# 9.3: Calculate the range of motion in cm for x axis

def compute_x_motion_per_finger(fused_hands):
    trajectories = {}
    for finger_idx in range(21):
        x_positions = [
            frame[finger_idx, 0] for frame in fused_hands if frame is not None
        ]
        if x_positions:
            motion_range = max(x_positions) - min(x_positions)
            trajectories[finger_idx] = motion_range
    return trajectories

# 9.4: Calculate wrist compensation ratio

def compute_wrist_compensation(fused_hands):
    wrist_movements = []
    finger_movements = []

    for frame in fused_hands:
        if frame is not None:
            wrist = frame[0, :2]
            fingertips = frame[[8, 12, 16, 20], :2]
            wrist_movements.append(np.linalg.norm(wrist))
            finger_avg = np.mean(np.linalg.norm(fingertips, axis=1))
            finger_movements.append(finger_avg)

    wrist_range = max(wrist_movements) - min(wrist_movements)
    finger_range = max(finger_movements) - min(finger_movements)

    return wrist_range / (finger_range + 1e-5)

# 9.5: Calculate amplitude (range of motion) in cm for x, y, z axe

def compute_amplitude(fused_hands):
    points = np.vstack([frame for frame in fused_hands if frame is not None])
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    amplitude = max_xyz - min_xyz
    return amplitude

# 9.6: Calculate smoothness (spectral arc length)

def compute_smoothness(fused_hands):
    from scipy.fft import fft
    positions = np.array([
        frame.mean(axis=0) for frame in fused_hands if frame is not None
    ])

    velocities = np.diff(positions, axis=0)
    speed = np.linalg.norm(velocities, axis=1)

    spectrum = np.abs(fft(speed))
    spectrum = spectrum[1:len(spectrum)//2]  # Remove DC component

    arc_length = np.sum(np.sqrt(1 + np.diff(spectrum)**2))
    smoothness = -arc_length  # Negative to follow spectral arc length convention
    return smoothness


# 9.7: Calculate finger independence

def compute_finger_independence(fused_hands):
    
    trajectories = {finger: [] for finger in finger_indices}

    for frame in fused_hands:
        if frame is None:
            continue
        for finger, idxs in finger_indices.items():
            mean_pos = frame[idxs, :].mean(axis=0)
            trajectories[finger].append(mean_pos)

    independences = {}
    for finger1 in finger_indices:
        corr_vals = []
        pos1 = np.array(trajectories[finger1])
        for finger2 in finger_indices:
            if finger1 == finger2:
                continue
            pos2 = np.array(trajectories[finger2])
            if len(pos1) == len(pos2) and len(pos1) > 1:
                corr = np.corrcoef(pos1[:, 0], pos2[:, 0])[0, 1]  # X-axis correlation
                corr_vals.append(abs(corr))
        independences[finger1] = np.mean(corr_vals)
    return independences

# 10. Process videos and visualize result

# Process and visualize healthy video
healthy_landmarks, healthy_scales = process_video(
    video_path=video_healthy_path,
    output_path=output_healthy_path,
    model_path=model_path,
    pt1=pt1, pt2=pt2, pt3=pt3, pt4=pt4, pt5=pt5, pt6=pt6
)

healthy_fused = compute_fused_hands(
    healthy_landmarks, healthy_scales, pt_left=pt1, pt_right=pt5
)

save_plots_and_animation(
    all_hands_landmarks=healthy_landmarks,
    fused_hands_all_frames_cm=healthy_fused,
    scales=healthy_scales,
    video_path=video_healthy_path,
    output_dir="/home/rosel/mnt/Hummel-Data/TiMeS/WP24/WP241/Kinematics/Code/healthy",
    pt_left=pt1,
    pt_right=pt5
)

# Process and visualize patient video
patient_landmarks, patient_scales = process_video(
    video_path=video_patient_path,
    output_path=output_patient_path,
    model_path=model_path,
    pt1=pt1, pt2=pt2, pt3=pt3, pt4=pt4, pt5=pt5, pt6=pt6
)

patient_fused = compute_fused_hands(
    patient_landmarks, patient_scales, pt_left=pt1, pt_right=pt5
)

save_plots_and_animation(
    all_hands_landmarks=patient_landmarks,
    fused_hands_all_frames_cm=patient_fused,
    scales=patient_scales,
    video_path=video_patient_path,
    output_dir="/home/rosel/mnt/Hummel-Data/TiMeS/WP24/WP241/Kinematics/Code/patient",
    pt_left=pt1,
    pt_right=pt5
)

### Compute Speeds ###
speeds_patient = compute_speeds(patient_fused, patient_scales['fps'])
speeds_healthy = compute_speeds(healthy_fused, healthy_scales['fps'])

# Average hand speeds per frame (for plotting)
avg_speed_patient = average_hand_speed(speeds_patient)
avg_speed_healthy = average_hand_speed(speeds_healthy)

# Finger speeds per frame (for plotting)
finger_speed_patient = finger_speeds(speeds_patient)
finger_speed_healthy = finger_speeds(speeds_healthy)

### Summary Statistics ###
# Whole-hand average speed over entire video
patient_hand_avg_speed = average_hand_speed_over_time(avg_speed_patient)
healthy_hand_avg_speed = average_hand_speed_over_time(avg_speed_healthy)

# Average speed per finger over video
patient_avg_finger_speeds = average_finger_speed_over_time(finger_speed_patient)
healthy_avg_finger_speeds = average_finger_speed_over_time(finger_speed_healthy)

# Max speed per finger
patient_max_speeds = max_finger_speed(finger_speed_patient)
healthy_max_speeds = max_finger_speed(finger_speed_healthy)

# Variability (std) per finger
patient_std_speeds = std_finger_speed(finger_speed_patient)
healthy_std_speeds = std_finger_speed(finger_speed_healthy)

# Axes of motion per finger
dominant_axis_patient, range_patient = compute_dominant_axis(patient_fused)
dominant_axis_healthy, range_healthy = compute_dominant_axis(healthy_fused)
x_motion_patient_raw = compute_x_motion_per_finger(patient_fused)
x_motion_healthy_raw = compute_x_motion_per_finger(healthy_fused)

# Convert from index to finger names
x_motion_patient = {finger: np.mean([x_motion_patient_raw[idx] for idx in indices])
                     for finger, indices in finger_indices.items()}

x_motion_healthy = {finger: np.mean([x_motion_healthy_raw[idx] for idx in indices])
                     for finger, indices in finger_indices.items()}

# Wrist compensation ratio
wrist_comp_patient = compute_wrist_compensation(patient_fused)
wrist_comp_healthy = compute_wrist_compensation(healthy_fused)

# Amplitude (range of motion) in cm
amplitude_patient = compute_amplitude(patient_fused)
amplitude_healthy = compute_amplitude(healthy_fused)

# Smoothness (spectral arc length)
smoothness_patient = compute_smoothness(patient_fused)
smoothness_healthy = compute_smoothness(healthy_fused)

# Finger independence
finger_independence_patient = compute_finger_independence(patient_fused)
finger_independence_healthy = compute_finger_independence(healthy_fused)

# Bar plot for finger independence
bar_plot_finger_summary(
    finger_independence_patient, finger_independence_healthy,
    "Finger Independence (Lower = More Independent)", "Correlation",
    "finger_independence.png", output_folder_patient
)

### 3. Print Summary ###
print("\n========= WHOLE HAND AVERAGE SPEED =========")
print(f"Patient: {patient_hand_avg_speed:.2f} cm/s")
print(f"Healthy: {healthy_hand_avg_speed:.2f} cm/s")

print("\n========= AVERAGE FINGER SPEED =========")
for finger in finger_indices.keys():
    print(f"{finger}: Patient {patient_avg_finger_speeds[finger]:.2f} cm/s | Healthy {healthy_avg_finger_speeds[finger]:.2f} cm/s")

print("\n========= MAX FINGER SPEED =========")
for finger in finger_indices.keys():
    print(f"{finger}: Patient {patient_max_speeds[finger]:.2f} cm/s | Healthy {healthy_max_speeds[finger]:.2f} cm/s")

print("\n========= FINGER SPEED VARIABILITY (STD) =========")
for finger in finger_indices.keys():
    print(f"{finger}: Patient {patient_std_speeds[finger]:.2f} cm/s | Healthy {healthy_std_speeds[finger]:.2f} cm/s")

print("\n========= DOMINANT AXIS OF MOVEMENT =========")
if dominant_axis_healthy is not None:
    print(f"Healthy movement range -> X: {range_healthy[0]:.2f} cm, Y: {range_healthy[1]:.2f} cm, Z: {range_healthy[2]:.2f} cm")
    print(f"Dominant axis: {dominant_axis_healthy}")
else:
    print("No valid data to compute dominant axis for healthy.")

if dominant_axis_patient is not None:
    print(f"Patient movement range -> X: {range_patient[0]:.2f} cm, Y: {range_patient[1]:.2f} cm, Z: {range_patient[2]:.2f} cm")
    print(f"Dominant axis: {dominant_axis_patient}")
else:
    print("No valid data to compute dominant axis for patient.")

print("\n========= X-AXIS MOTION PER FINGER =========")
for finger in finger_indices.keys():
    print(f"{finger}: Patient {x_motion_patient[finger]:.2f} cm | Healthy {x_motion_healthy[finger]:.2f} cm")

print("\n========= WRIST COMPENSATION RATIO =========")
print(f"Patient: {wrist_comp_patient:.3f}")
print(f"Healthy: {wrist_comp_healthy:.3f}")

print("\n========= AMPLITUDE (RANGE OF MOTION) =========")
print(f"Patient (X, Y, Z): {amplitude_patient}")
print(f"Healthy (X, Y, Z): {amplitude_healthy}")

print("\n========= SMOOTHNESS (SPECTRAL ARC LENGTH) =========")
print(f"Patient: {smoothness_patient:.3f}")
print(f"Healthy: {smoothness_healthy:.3f}")

print("\n========= FINGER INDEPENDENCE =========")
for finger in finger_indices.keys():
    print(f"{finger}: Patient {finger_independence_patient[finger]:.3f} | Healthy {finger_independence_healthy[finger]:.3f}")

bar_plot_finger_summary(
    patient_values=finger_independence_patient,
    healthy_values=finger_independence_healthy,
    title="Finger Independence (Lower = More Independent)",
    ylabel="Correlation",
    filename="finger_independence.png",
    output_folder=output_folder_patient  # or healthy, depending on target
)


### Generate Plots ###
# Hand speed over time
plot_avg_hand_speed_over_time(avg_speed_patient, avg_speed_healthy, output_folder_patient)

# Finger speeds over time
plot_finger_speeds_over_time(finger_speed_patient, finger_speed_healthy, output_folder_patient)

# Summary bars
bar_plot_finger_summary(patient_avg_finger_speeds, healthy_avg_finger_speeds,
                         'Average Finger Speed', 'Speed (cm/s)', 'avg_finger_speed.png', output_folder_patient)

bar_plot_finger_summary(patient_max_speeds, healthy_max_speeds,
                         'Maximum Finger Speed', 'Speed (cm/s)', 'max_finger_speed.png', output_folder_patient)

bar_plot_finger_summary(patient_std_speeds, healthy_std_speeds,
                         'Finger Speed Variability (STD)', 'STD (cm/s)', 'std_finger_speed.png', output_folder_patient)

print("\nAll plots saved in folder.")