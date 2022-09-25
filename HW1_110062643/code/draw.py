import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(main_path, 'data', 'dog.bmp')
    image_BGR = cv2.imread(img_path)
    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    img_cv2_ori = cv2.imread(img_path)
    f = plt.figure('image_BGR vs. image_RGB')
    f.add_subplot(1,3, 1)
    plt.imshow(img_cv2_ori)
    plt.axis('off')
    plt.title('img_cv2_ori')
    f.add_subplot(1, 3, 2)   # 生成1 row, 2 column，然後取第一個
    plt.imshow(image_BGR)
    plt.axis('off')
    plt.title('image_BGR')
    f.add_subplot(1, 3, 3)
    plt.imshow(image_RGB)
    plt.axis('off')
    plt.title('image_RGB')
    plt.savefig('tmp.jpg')
