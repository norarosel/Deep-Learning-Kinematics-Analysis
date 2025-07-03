#Nora Cécile Rosel Zaballos - 04/06/2025 - First attempt at using py and server
#Using mediapipe.solutions.hands, probably works too but it's older
import cv2
import mediapipe as mp
import os

print("Current working directory:", os.getcwd())

# Initialize MediaPipe drawing and hands modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Video file path (utube video)
video_path = "./video.mp4"

# Opening video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video file:", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video FPS:", fps)
print("Frame size:", (width, height))

# Define codec and create VideoWriter object 
output_path = "/home/rosel/Documents/annotated_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("ERROR: VideoWriter failed to open.")
    exit()

#Create a Hands object with specific settings, it takes the webcam input code
#Check HandLandmarker VIDEO instead
#Check the static images to download anotated images, check how to visualize the video
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        success, image = cap.read()
        if not success or image is None:
            print("End of video or failed to read frame.")
            break

        print("Image shape:", image.shape)
        #To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        #Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Print all 21 landmarks in (x, y, z)
                print(f"\nFrame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} - Hand {hand_idx + 1}:")
                for i, lm in enumerate(hand_landmarks.landmark):
                    print(f"  Landmark {i}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f}")
                
                # Optionally draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


                out.write(image)  # Write the annotated frame to output video

cap.release()

print("Processing complete. Output saved to:", output_path)
