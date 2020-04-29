from utils.image import random_visual_effect_generator
import cv2
img = cv2.imread('test/demo.jpg')
aug = random_visual_effect_generator()
while True:
    gen = next(aug)(img)
    print(type(gen))
    cv2.imshow("trest", gen)
    if cv2.waitKey() == 27:
        break
    cv2.destroyAllWindows()