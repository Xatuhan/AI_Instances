import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Button, Label, messagebox

def imread_unicode(path: str):
    #Dosya var mı yok mu türkçe harfli yola sahip mi kontrolü
    try:
        with open(path, "rb") as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None

def select_image():
    filepath = filedialog.askopenfilename(
        title="Resim seç",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
    )
    if not filepath:
        return


    bgr = cv2.imread(filepath)


    if bgr is None:
        bgr = imread_unicode(filepath)

    if bgr is None:
        messagebox.showerror("Hataa", f"Görüntü okunamadı:\n{filepath}\n\nDosya yolu kontrol et.")
        return

    segment_image_watershed(bgr)

def segment_image_watershed(bgr):


    img = bgr.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    mean_val = gray.mean()
    thresh_type = cv2.THRESH_BINARY_INV if mean_val > 127 else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown
    unknown = cv2.subtract(sure_bg, sure_fg)

    #  Marker
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

   # Watershed
    markers = cv2.watershed(img, markers)

    # Görüntünün sınırlarını renkleme
    overlay = img.copy()
    overlay[markers == -1] = (0, 0, 255)


    mask = markers.copy()
    mask[mask <= 1] = 0          # arka plan ve unknown
    mask[mask == -1] = 0         # sınır

    vals, cnts = np.unique(mask, return_counts=True)

    # 0 arka plan gibi düşün (biz zaten 0 yaptık ama garanti olsun)
    valid = vals != 0
    vals = vals[valid]
    cnts = cnts[valid]

    if len(vals) > 0:
        largest_label = vals[np.argmax(cnts)]
        binary_mask = (markers == largest_label).astype(np.uint8) * 255
    else:
        binary_mask = np.zeros(gray.shape, dtype=np.uint8)

    display_segmented_images(img, overlay, binary_mask)

def display_segmented_images(original_bgr, overlay_bgr, mask):
    cv2.imshow("Orijinal", original_bgr)
    cv2.imshow("Watershed Formu", overlay_bgr)
    cv2.imshow("Mask (Binary)", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#  Tkinter UI Pythonda çalışan basit ui için 
app = tk.Tk()
app.title("Image Segmentation Tool (Watershed)")

label = Label(app, text="Bir resim seç: Watershed ile piksel bazlı segmentasyon yapılacak.")
label.pack(pady=10)

select_button = Button(app, text="Select Image", command=select_image)
select_button.pack(pady=10)

app.geometry("380x160")
app.mainloop()
