import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ['Deepfake', 'Original']

# Load the pre-trained model
model_path = 'E:\deepfake dataset\convlstm_model_Date_Time2025_02_23__09_41_04__Loss0.40521931648254395__Accuracy_0.8235294222831726.h5'
convlstm_model = load_model(model_path)

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH, model, CLASSES_LIST):
    # Initialize video reader and get properties
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        output_file_path,
        cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
        fps,
        (original_video_width, original_video_height)
    )

    # Initialize a queue to hold frames for prediction
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        # Make predictions only when the queue is full
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_label_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_label_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            #predicted_class_name = 'original' if CLASSES_LIST[predicted_label] == 'deepfake' else 'deepfake'
            # Display prediction on the frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # Write the frame to the output video
        video_writer.write(frame)

    # Release resources
    video_reader.release()
    video_writer.release()

    print(f"Output video saved at: {output_file_path}")

def main():
    # Get user input for the video file path
    input_video_file_path = input("Enter the path to the video file: ").strip()
    
    # Check if the file exists
    if not os.path.exists(input_video_file_path):
        print("Error: The specified video file does not exist.")
        return

    # Define the output video path
    output_video_file_path = os.path.join('output_videos', os.path.basename(input_video_file_path))
    os.makedirs('output_videos', exist_ok=True)

    # Perform prediction on the video
    predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH, convlstm_model, CLASSES_LIST)

    print(f"Processing complete. Output video saved at: {output_video_file_path}")

if __name__ == "__main__":
    main()