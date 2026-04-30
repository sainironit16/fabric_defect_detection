import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("fabric_model2.h5", compile=False)

def predict_on_image(img_bgr, window_name="Result"):
    rgb_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(rgb_frame, (224, 224))
    
    # MobileNetV2 preprocessing
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    pred = model.predict(img_preprocessed, verbose=0)[0][0]

    # --- CALIBRATION THRESHOLD ---
    # Industrial fabric models can be overly sensitive to shadows or folds.
    # We increase the required confidence threshold from 0.50 to 0.95.
    # Scores > 0.95 = Defect. Scores <= 0.95 = Normal. 
    THRESHOLD = 0.95

    if pred > THRESHOLD:
        label = "DEFECT"
        color = (0, 0, 255)
    else:
        label = "NORMAL"
        color = (0, 255, 0)

    result_frame = img_bgr.copy()
    
    # Scale text size based on image resolution so it's always readable
    height, width = result_frame.shape[:2]
    font_scale = max(1, width // 600)
    thickness = max(2, width // 300)
    
    cv2.putText(result_frame, label, (50, 50 * font_scale),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # If the image is extremely large, restrict the window size
    if width > 1200 or height > 900:
        cv2.resizeWindow(window_name, width // 2, height // 2)
        
    cv2.imshow(window_name, result_frame)
    return result_frame

def main():
    print("===============================")
    print(" Fabric Anomaly Detection")
    print("===============================")
    print("1. Use External Webcam")
    print("2. Upload/Provide an Image File")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        # 1 is usually the external webcam, 0 is the built-in laptop camera
        print("Initializing External Webcam...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened() or not cap.read()[0]:
            print("External webcam not found at index 1. Falling back to built-in webcam (index 0)...")
            cap = cv2.VideoCapture(0)

        print("\nPress 'c' to capture a picture and detect anomaly.")
        print("Press 'q' or ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            # Show live feed
            live_frame = frame.copy()
            cv2.putText(live_frame, "Press 'c' to capture, 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Fabric Detection - Live Feed", live_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                # Process the captured frame
                result_frame = predict_on_image(frame, "Result")
                
                cv2.putText(result_frame, "Press any key to resume live feed", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Result", result_frame)
                cv2.waitKey(0) # Wait indefinitely
                cv2.destroyWindow("Result")

        cap.release()
        cv2.destroyAllWindows()
        
    elif choice == '2':
        img_path = input("Enter the filename (e.g., test.jpg): ").strip()
        if not os.path.exists(img_path):
            print(f"Error: File '{img_path}' not found in the current folder.")
        else:
            frame = cv2.imread(img_path)
            if frame is None:
                print("Error: Could not read the image. It might be corrupted or an unsupported format.")
            else:
                predict_on_image(frame, "Fabric Detection - Image")
                print("Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")

if __name__ == "__main__":
    main()