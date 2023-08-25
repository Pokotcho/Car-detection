import cv2
import numpy as np
from time import sleep

offset = 6

pos_line = 600

delay = 60

min_car = 55
max_car = 150

detec = []
Cars = 0


def Vehicule_entry(x, y):
    cv2.line(frame1, (200, pos_line), (1050, pos_line), (0, 255, 255), 3)
    detec.remove((x, y))


def Center_Point(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("video.mp4")
substraction = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    useless, original = cap.read()
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = substraction.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (200, pos_line), (1050, pos_line), (255, 0, 0), 3)
    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validation_contours = (min_car < w < max_car) and (min_car < h < max_car)
        if not validation_contours:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = Center_Point(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        if (min_car < w < max_car) or (min_car < h < max_car):
            cv2.putText(
                frame1,
                "Car",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 255, 0),
                2,
            )
        else:
            continue

        for x, y in detec:
            if (
                y < (pos_line + offset)
                and y > (pos_line - offset)
                and x < (1050 + offset)
                and ((min_car < w < max_car) or (min_car < h < max_car))
            ):
                Cars += 1
                Vehicule_entry(x, y)
                print("Total Cars : " + str(Cars))

    cv2.putText(
        frame1,
        "Cars in the parking : " + str(Cars),
        (460, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    cv2.imshow("Original Video", frame1)
    # cv2.imshow("Detector", dilated)
    # cv2.imshow("Original", original)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
