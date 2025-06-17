import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

# This is the standard set of connections from MediaPipe
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# --- THE ONLY CHANGE IS HERE ---
# Instead of using 0 for the webcam, we provide the path to a video file.
# Make sure your video file is in the same folder as this script.
video_file_name = 'my_dance.mp4'  # <--- CHANGE THIS TO YOUR FILENAME
cap = cv2.VideoCapture(video_file_name)
# -----------------------------

# --- 3D PLOT SETUP ---
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # Interactive mode ON

# --- COLOR & TIMING SETUP ---
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
color_index = 0
last_color_change = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    # If `ret` is False, it means we have reached the end of the video.
    if not ret:
        print("End of video. Exiting...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_count += 1

    if time.time() - last_color_change > 3:
        color_index = (color_index + 1) % len(colors)
        last_color_change = time.time()
    
    glow_color = colors[color_index]

    h, w, _ = frame.shape
    clone_canvas = np.zeros_like(frame, dtype=np.uint8)

    if results.pose_landmarks:
        # --- 2D CLONE DRAWING ---
        landmarks = results.pose_landmarks
        
        mp.solutions.drawing_utils.draw_landmarks(
            clone_canvas, landmarks, POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=glow_color, thickness=10),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=5)
        )
        
        glow = cv2.GaussianBlur(clone_canvas, (51, 51), 0)
        clone_with_glow = cv2.addWeighted(clone_canvas, 1.0, glow, 0.7, 0)

        # --- 3D PLOT UPDATE (MANUAL PLOTTING) ---
        if frame_count % 5 == 0:
            ax.clear()
            ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
            ax.view_init(elev=10., azim=-90)
            ax.set_title("3D Pose Visualization")
            
            landmarks_3d = results.pose_landmarks.landmark
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_point, end_point = landmarks_3d[start_idx], landmarks_3d[end_idx]
                ax.plot([start_point.x, end_point.x],
                        [-start_point.y, -end_point.y],
                        [-start_point.z, -end_point.z],
                        color='cyan')

            plt.draw()
            plt.pause(0.001)

    else:
        clone_with_glow = clone_canvas

    combined_output = np.hstack((frame, clone_with_glow))
    cv2.imshow("Dance Clone - Video File", combined_output)

    # Use waitKey(1) for video files to process them as fast as possible.
    # Press 'q' to quit early.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()