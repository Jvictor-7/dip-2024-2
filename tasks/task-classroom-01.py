import argparse
import numpy as np
import cv2 as cv
import requests

# -- Function to load image from URL -- #
def load_image_from_url(url, **kwargs):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv.imdecode(image_array, kwargs.get("flags", cv.IMREAD_COLOR)) 
    
    return image

# -- Function to resize image keeping the ratio -- #
def resize_keep_ratio(image, height, width):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

# -- Function to combine images -- #
def combinationImages(*args):
    a = cv.getTrackbarPos("a", "Slides") / 100
    b = cv.getTrackbarPos("b", "Slides") / 100
    
    # linear combination:  h = af + bg
    h = (a * f + b * g).clip(0, 255)
    
    h = h.astype(np.uint8)
    
    cv.imshow("Combinacao", h)
    
    a_text = f"a: {a:.2f}"
    b_text = f"b: {b:.2f}"
    
    cv.putText(h, a_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv.putText(h, b_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    cv.imshow("Combinacao", h)
    
# -- Main -- #
f = load_image_from_url("https://blog-static.petlove.com.br/wp-content/uploads/2024/04/29030254/cachorro-vira-lata-Petlove.jpg")
g = load_image_from_url("https://www.coa.com.br/wp-content/uploads/2023/12/COA-3-1.png")

target_width, target_height = 500, 500
f = resize_keep_ratio(f, target_width, target_height)
g = resize_keep_ratio(g, target_width, target_height)

cv.namedWindow('Slides')
cv.resizeWindow('Slides', 500, 80)

cv.createTrackbar("a", "Slides", 50, 100, combinationImages) 
cv.createTrackbar("b", "Slides", 50, 100, combinationImages)

cv.imshow("Imagem F", f)   
cv.imshow("Imagem G", g)  

cv.moveWindow('Imagem F', 0, 0)
cv.moveWindow('Imagem G', 500, 0)
cv.moveWindow('Slides', 0, 500)
cv.moveWindow("Combinacao", 500, 400)

cv.waitKey(0)

cv.destroyAllWindows()