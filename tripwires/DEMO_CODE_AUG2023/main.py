import time
import math
import cv2
import numpy as np
import os

import cv2

def sobel_n(n, image_gray):

    if n > 0:
        grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        # grad_x = 0
        grad = np.sqrt(grad_y**2+grad_x**2)
        grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

        n -= 1
        return sobel_n(n, grad_norm)
    return image_gray

#
def resize(percent, img):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def nothing(x):
    pass

def lines_detection(mask, img, num_of_inters,  min_lenght, max_gap):

    lines_p = cv2.HoughLinesP(mask, 1, np.pi / 180, num_of_inters, None, min_lenght, max_gap)
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            angle = math.atan2(line[3] - line[1], line[2] - line[0])*180 / np.pi
            if abs(angle) < 10:
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv2.LINE_AA)
    return img


def toolbar(x, y):

    cv2.namedWindow('Toolbar window')
    cv2.resizeWindow('Toolbar window', 600, 400)
    cv2.createTrackbar('Scan_w', 'Toolbar window', int(x/4), int(x/2), nothing)
    cv2.createTrackbar('Scan_h', 'Toolbar window', int(y/4), y, nothing)
    cv2.createTrackbar('Canny_L', 'Toolbar window', 100, 255, nothing)
    cv2.createTrackbar('Canny_U', 'Toolbar window', 200, 255, nothing)
    cv2.createTrackbar('num_of_inters', 'Toolbar window',50, 500, nothing)
    cv2.createTrackbar('min_lenght', 'Toolbar window', 200, 500, nothing)
    cv2.createTrackbar('max_gap', 'Toolbar window', 50, 200, nothing)


def working_area():
    pass


if __name__ == '__main__':
    toolbar_flag = True
    representing_val = 50
    camera_id = 0
    
    cap = cv2.VideoCapture(camera_id)  # this is the magic!

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print( frame.shape)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here

        img_show = resize(representing_val, frame)
        scan_y_max, scan_x_max, _ = img_show.shape
        if toolbar_flag:
            toolbar(scan_x_max, scan_y_max)
            toolbar_flag = False

        w_point = cv2.getTrackbarPos('Scan_w', 'Toolbar window')
        h_point = cv2.getTrackbarPos('Scan_h', 'Toolbar window')

        scale_to_full = 100/representing_val

        start_y = int((scan_y_max-h_point) * scale_to_full)
        end_y = int(scan_y_max * scale_to_full)

        start_x = int((scan_x_max/2 - w_point) * scale_to_full)
        end_x = int((scan_x_max/2 + w_point) * scale_to_full)

        if start_y-end_y == 0 or start_x-end_x == 0:
            print('no area for scanning')
            cropped_image = frame
        else:
            cropped_image = frame[start_y: end_y, start_x: end_x]

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        canny_lower = cv2.getTrackbarPos('Canny_L', 'Toolbar window')
        canny_upper = cv2.getTrackbarPos('Canny_U', 'Toolbar window')
        num_of_inters = cv2.getTrackbarPos('num_of_inters', 'Toolbar window')
        min_lenght = cv2.getTrackbarPos('min_lenght', 'Toolbar window')
        max_gap = cv2.getTrackbarPos('max_gap', 'Toolbar window')

        edges = cv2.Canny(gray, canny_lower, canny_upper)

        lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, num_of_inters, None, min_lenght, max_gap)
        if lines_p is not None:
            for i in range(0, len(lines_p)):
                line = lines_p[i][0]
                angle = math.atan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi
                if abs(angle) < 20:
                    cv2.line(cropped_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10, cv2.LINE_AA)
        cropped_image = resize(representing_val, cropped_image)


        overlay = img_show.copy()
        if start_y - end_y == 0 or start_x - end_x == 0:
            print('no area for scanning')
            img_show = cropped_image
        else:
            img_show[scan_y_max-h_point: scan_y_max, int(scan_x_max//2 - w_point): int(scan_x_max//2 + w_point)] = cropped_image

        cv2.line(img_show, (scan_x_max // 2 - w_point, scan_y_max), (scan_x_max // 2 - w_point, scan_y_max - h_point),
                 (255, 0, 0), 5)
        cv2.line(img_show, (scan_x_max // 2 + w_point, scan_y_max), (scan_x_max // 2 + w_point, scan_y_max - h_point),
                 (0, 255, 0), 5)

        result = cv2.addWeighted(overlay, 0.5, img_show, 0.5, 0)

        cv2.imshow('Image window',  result)
        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

