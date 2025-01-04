import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load model architecture from JSON
try:
    with open('./new_model/best_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    print("Model architecture loaded successfully.")
except FileNotFoundError:
    print("Error: 'inceptionv3_Transfer_Learning.json' file not found.")
    exit()

# Load model weights
try:
    model.load_weights("./new_model/best_model.weights.h5")
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Error: 'custom_cnn_augmentation_model.weights.h5' file not found.")
    exit()


# Function to preprocess the image for the model
def preprocess_image(image):
    try:
        # Resize image to match the model's input size (224x224 for InceptionV3)
        img_resized = cv2.resize(image, (224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_input(img_array)  # Normalize input for InceptionV3
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


# Function to capture and predict
def capture_and_predict():
    # Start video capture
    cap = cv2.VideoCapture(1)  # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 's' to capture an image or 'q' to quit.")

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Display the captured frame
        cv2.imshow("Webcam Feed", frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to capture and predict
            print("Capturing and processing the image...")
            preprocessed_image = preprocess_image(frame)

            if preprocessed_image is not None:
                # Perform prediction
                predictions = model.predict(preprocessed_image)
                print(f"Raw Predictions: {predictions}")

                # Add post-processing to interpret predictions
                predicted_class = np.argmax(predictions, axis=1)
                print(f"Predicted Class: {predicted_class}")
            else:
                print("Error: Preprocessed image is invalid.")

        elif key == ord('q'):  # Press 'q' to quit
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Run the capture and prediction function
if __name__ == "__main__":
    capture_and_predict()
