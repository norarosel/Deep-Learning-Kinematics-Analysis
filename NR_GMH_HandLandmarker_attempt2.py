# Nora Cécile Rosel Zaballos - 05/06/2025 - Now with HandLandmarker
# 13/06/2025: Box + 3D Animation

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

# 1. SETUP: Define file paths
video_path = "/home/rosel/mnt/Hummel-Data/TiMeS/WP24/WP241/Kinematics/Videos/patient_1.mp4"
output_path = "/home/rosel/mnt/Hummel-Data/TiMeS/WP24/WP241/Kinematics/Code/annotated_output_patient.mp4"
model_path = "/home/rosel/mnt/Hummel-Data/TiMeS/WP24/WP241/Kinematics/Code/hand_landmarker.task"

# 2. Load the input video
print(f"Loading video from: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS: {fps}, Resolution: {width}x{height}")

# 3. Set up the output video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()

# 4. Initialize MediaPipe hand detection model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Coordinates for custom lines (pixel coordinates)
pt1 = (202, 515)
pt2 = (1541, 774)
pt3 = (367, 795)
pt4 = (196, 796)
pt5 = (1247, 864)
pt6 = (1549, 865)

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

# 5. Run detection and save results
all_hands_landmarks = []

print("Initializing HandLandmarker...")
with HandLandmarker.create_from_options(options) as landmarker:
    print("HandLandmarker initialized.")
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Finished processing video.")
            break

        print(f"\nProcessing frame {frame_index}")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int((frame_index / fps) * 1000)

        detection_result = landmarker.detect_for_video(mp_image, timestamp)

        frame_landmarks = []
        if detection_result.hand_landmarks:
            print(f"Detected {len(detection_result.hand_landmarks)} hand(s)")
            for hand_idx, landmarks_list in enumerate(detection_result.hand_landmarks):
                single_hand = [(lm.x, lm.y, lm.z) for lm in landmarks_list]
                frame_landmarks.append(single_hand)

                converted_landmarks = [
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks_list
                ]
                landmark_proto = landmark_pb2.NormalizedLandmarkList(landmark=converted_landmarks)

                mp_drawing.draw_landmarks(
                    frame,
                    landmark_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i, lm in enumerate(landmarks_list):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            print("No hands detected.")

        # Draw red lines between the specified points
        cv2.line(frame, pt1, pt4, (0, 0, 255), thickness=2)
        cv2.line(frame, pt4, pt3, (0, 0, 255), thickness=2)
        cv2.line(frame, pt5, pt6, (0, 0, 255), thickness=2)
        cv2.line(frame, pt6, pt2, (0, 0, 255), thickness=2)

        all_hands_landmarks.append(frame_landmarks)
        out.write(frame)
        frame_index += 1

cap.release()
out.release()
print("\nAnnotated video saved to:", output_path)

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

# Extract hands frame 0 for plotting and fusion
if len(all_hands_landmarks) > 0 and len(all_hands_landmarks[0]) == 2:
    hand1_norm = all_hands_landmarks[0][0]
    hand2_norm = all_hands_landmarks[0][1]

    hand1_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand1_norm]
    hand2_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand2_norm]

    # Convert to cm with real-world x,y from hand1 reference pt5, z scaled approx for hand1
    hand1_cm = convert_hand_to_cm(hand1_px, scale_x_right, scale_y_right, scale_z_right, pt5)
    # Convert to cm with real-world x,y from hand2 reference pt1, z scaled approx for hand2
    hand2_cm = convert_hand_to_cm(hand2_px, scale_x_left, scale_y_left, scale_z_left, pt1)

    # Fuse z by averaging mediapipe z's (index matched)
    fused_hand_cm = []
    for i in range(len(hand1_cm)):
        x = hand1_cm[i, 0]  # Use x,y from hand1 in cm (as requested)
        y = hand1_cm[i, 1]
        z = (hand1_cm[i, 2] + hand2_cm[i, 2]) / 2
        fused_hand_cm.append([x, y, z])
    fused_hand_cm = np.array(fused_hand_cm)
else:
    print("Not enough hands detected in frame 0 for fusion.")
    exit()

fused_hands_all_frames_cm = []

for frame_landmarks in all_hands_landmarks:
    if len(frame_landmarks) == 2:
        hand1_norm = frame_landmarks[0]
        hand2_norm = frame_landmarks[1]

        hand1_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand1_norm]
        hand2_px = [(lm[0] * width, lm[1] * height, lm[2]) for lm in hand2_norm]

        hand1_cm = convert_hand_to_cm(hand1_px, scale_x_right, scale_y_right, scale_z_right, pt5)
        hand2_cm = convert_hand_to_cm(hand2_px, scale_x_left, scale_y_left, scale_z_left, pt1)

        fused_cm = []
        for i in range(len(hand1_cm)):
            x = hand1_cm[i, 0]
            y = hand1_cm[i, 1]
            z = (hand1_cm[i, 2] + hand2_cm[i, 2]) / 2
            fused_cm.append([x, y, z])
        fused_hands_all_frames_cm.append(np.array(fused_cm))
    else:
        fused_hands_all_frames_cm.append(None)


# Plot Hand 1 frame 0 (real-world coords)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
plot_hand_3d(ax, hand1_cm, 'red', 'Hand 1 Frame 0')
ax.set_title("Hand 1 Frame 0 (Real World Coordinates)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.view_init(elev=30, azim=120)
plt.tight_layout()
plt.savefig("hand1_frame0_real_world.png")
plt.close(fig)

# Plot Hand 2 frame 0 (real-world coords)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
plot_hand_3d(ax, hand2_cm, 'blue', 'Hand 2 Frame 0')
ax.set_title("Hand 2 Frame 0 (Real World Coordinates)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.view_init(elev=30, azim=120)
plt.tight_layout()
plt.savefig("hand2_frame0_real_world.png")
plt.close(fig)

# Plot Fused Hand frame 0
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
plot_hand_3d(ax, fused_hand_cm, 'green', 'Fused Hand Frame 0')
ax.set_title("Fused Hand Frame 0 (Real World Coordinates)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")
ax.view_init(elev=40, azim=120)
plt.tight_layout()
plt.savefig("fused_hand_frame0.png")
plt.close(fig)


# Prepare fused hands in cm for animation over first 15 frames (filter out None)
frames_to_use = [f for f in fused_hands_all_frames_cm[:100] if f is not None]

if len(frames_to_use) == 0:
    print("No valid fused hand frames available for animation.")
    exit()

# Use first valid frame for limits
fused_hand_frame0_cm = frames_to_use[0]

x_min, x_max = fused_hand_frame0_cm[:, 0].min(), fused_hand_frame0_cm[:, 0].max()
y_min, y_max = fused_hand_frame0_cm[:, 1].min(), fused_hand_frame0_cm[:, 1].max()
z_min, z_max = fused_hand_frame0_cm[:, 2].min(), fused_hand_frame0_cm[:, 2].max()

# Add some padding for better visualization
pad = 1.0  # cm
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

# Animate only first 15 frames of fused cm data
anim = FuncAnimation(fig_anim, update_anim, frames=len(frames_to_use), init_func=init_anim, blit=False)
anim.save("fused_hand_animation_15frames_cm.mp4", fps=15)
plt.close(fig_anim)

print("Saved fixed-limits animation video: fused_hand_animation_15frames_cm.mp4")




# --- Simple Kinematic Analysis ---
# Calculate instantaneous speed of each landmark over the 100 frames (cm/s)
# speed = distance between frames * fps

speeds = []
for i in range(len(fused_hands_all_frames_cm[0])):  # iterate over landmarks
    landmark_speeds = []
    for f in range(1, 100):
        if fused_hands_all_frames_cm[f] is None or fused_hands_all_frames_cm[f-1] is None:
            landmark_speeds.append(0)
            continue
        p1 = fused_hands_all_frames_cm[f-1][i]
        p2 = fused_hands_all_frames_cm[f][i]
        dist = np.linalg.norm(p2 - p1)  # Euclidean distance in cm
        speed = dist * fps  # cm/s
        landmark_speeds.append(speed)
    speeds.append(np.mean(landmark_speeds))

# Find landmark with max average speed
max_speed = max(speeds)
max_idx = speeds.index(max_speed)

print(f"\nKinematic Analysis:")
print(f"Landmark {max_idx} moves fastest with average speed: {max_speed:.2f} cm/s")

# Calculate average displacement vector direction for that landmark over frames
vec_sum = np.zeros(3)
count = 0
for f in range(1, 100):
    if fused_hands_all_frames_cm[f] is None or fused_hands_all_frames_cm[f-1] is None:
        continue
    p1 = fused_hands_all_frames_cm[f-1][max_idx]
    p2 = fused_hands_all_frames_cm[f][max_idx]
    vec_sum += (p2 - p1)
    count += 1

if count > 0:
    avg_dir = vec_sum / count
    norm = np.linalg.norm(avg_dir)
    if norm > 0:
        avg_dir /= norm
else:
    avg_dir = np.array([0, 0, 0])

print(f"Average movement direction of landmark {max_idx} (unit vector): {avg_dir}")

# Interpret direction: which axis dominates?
dominant_axis = np.argmax(np.abs(avg_dir))
axis_names = ['X (left-right)', 'Y (up-down)', 'Z (depth)']
direction = "positive" if avg_dir[dominant_axis] > 0 else "negative"

print(f"Dominant movement axis: {axis_names[dominant_axis]}, direction: {direction}")

print("\nInterpretation:")
print(f"Landmark {max_idx} is moving at ~{max_speed:.2f} cm/s mostly along {axis_names[dominant_axis]} axis in the {direction} direction.")






