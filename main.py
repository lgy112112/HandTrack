import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    points = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id in [0, 5, 17]:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
    if len(points) == 3:
        centerX = int((points[0][0] + points[1][0] + points[2][0]) / 3)
        centerY = int((points[0][1] + points[1][1] + points[2][1]) / 3)
        cv.circle(img, (centerX, centerY), 5, (255, 0, 0), cv.FILLED)

        # 绘制一个框
        box_size = 200  # 框的大小
        top_left = (centerX - int(box_size / 2), centerY - int(box_size / 2))
        bottom_right = (centerX + int(box_size / 2), centerY + int(box_size / 2))
        cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # 提取框内图像区域
        cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # 检查图像区域是否为空
        if cropped_img.size > 0:
            # 显示框内图像区域
            cv.imshow("Cropped Image", cropped_img)
    else:
        # 如果没有检测到手部关键点，则显示原始图像
        cv.imshow("Image", img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # img = cv.resize(img, (1080, 720))

    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()