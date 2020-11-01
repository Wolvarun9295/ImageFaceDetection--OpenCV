import cv2

try:
    faces = cv2.CascadeClassifier(cv2.haarcascades + "haarcascade_frontalface_alt.xml")

    img = cv2.imread("images/ironman.jpeg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = faces.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=6)

    print(f"Possible face detected at: {detections}")

    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

    ims = cv2.resize(img, (600, 800))

    cv2.imshow("output", ims)

    cv2.waitKey(0)

    # cv2.imwrite("images/output.jpg", ims)

except:
    print("Something went wrong!")
