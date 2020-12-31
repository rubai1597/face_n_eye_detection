import cv2
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    for file_name in Path("./img").glob("*.jpg"):
        img = cv2.imread(str(file_name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        for x, y, w, h in faces:
            face_gray = img_gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow("Eye detection", img)
        cv2.waitKey(0)
        cv2.imwrite(f"./img/result_{file_name.name}", img)
