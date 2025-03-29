import cv2
from mtcnn import MTCNN

video_path = 0  # Change this to your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize MTCNN face detector
detector = MTCNN()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends
    
    # Convert frame to RGB (MTCNN requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)
    
    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Write the frame to output video
    out.write(frame)
    
    # Display the frame (optional)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Output video saved at:", output_path)
