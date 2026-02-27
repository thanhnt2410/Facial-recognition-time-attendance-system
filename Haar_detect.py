import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Mở camera (Windows nên thêm CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize cho nhẹ hơn
    frame_small = cv2.resize(frame, (640, 480))

    # Convert sang grayscale
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Vẽ khung
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Haar Face Detection", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
