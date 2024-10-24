import cv2

# Load the face and smile cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Define the font for displaying text on the image
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the smile counter and the image capture flagrr
smile_count = 0
capture_image = False

while True:

    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, draw a rectangle around it and look for smiles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # For each smile detected, draw a rectangle around it and capture the image
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            smile_count += 1
            if smile_count >= 10:  # Change the smile count  as needed
                capture_image = True

    if capture_image:
        cv2.imwrite('captured_image.jpg', frame)
        cv2.putText(frame, "Image captured!", (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        smile_count = 0
        capture_image = False

    # Display the image
    cv2.imshow('Smile Detector', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()