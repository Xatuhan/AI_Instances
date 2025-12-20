import cv2
import numpy as np

def largest_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Kamera açılamadı.")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #  Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    #  Maske temizleme
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnt = largest_contour(th)
    mask = np.zeros_like(th)

    if cnt is not None and cv2.contourArea(cnt) > 1500:
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)  # doldur => segmentation mask
        # Kontur çizimi (görsel için)
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        segmented = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        segmented = np.zeros_like(frame)

    cv2.imshow("Live", frame)
    cv2.imshow("Mask (Binary)", mask)
    cv2.imshow("Segmented", segmented)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()



