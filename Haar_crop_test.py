import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        padding = 25

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame_small.shape[1])
        y2 = min(y + h + padding, frame_small.shape[0])

        face_crop = frame_small[y1:y2, x1:x2]
        face_resize = cv2.resize(face_crop, (160, 160))

        # # Vẽ khung
        # cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop mặt
        face_crop = frame_small[y:y+h, x:x+w]

        # Resize đúng chuẩn FaceNet
        face_resize = cv2.resize(face_crop, (160, 160))

        # Hiển thị cửa sổ mặt riêng
        cv2.imshow("Cropped Face (160x160)", face_resize)

    cv2.imshow("Haar Detection", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
