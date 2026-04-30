import cv2
import numpy as np
import onnxruntime as ort

THRESHOLD = 0.95

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Running... Press Ctrl+C to stop")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame error")
            break

        img = cv2.resize(frame, (224,224))
        img = img.astype('float32')
        img = (img / 127.5) - 1.0

        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0,3,1,2))

        output = session.run(None, {input_name: img})
        pred = output[0][0][0]

        if pred > THRESHOLD:
            label = "DEFECT"
            color = (0, 0, 255)
        else:
            label = "NORMAL"
            color = (0, 255, 0)

        cv2.putText(frame, label, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Fabric Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopped")

cap.release()
cv2.destroyAllWindows()