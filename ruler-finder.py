# -*- coding: utf-8 -*-


import cv2
import numpy as np
import imutils


# меняем цвет на оттенки серого и применяем резкость
image = cv2.imread("example3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imwrite("gray.jpg", gray)
# определяем края (ищем контуры)
edged = cv2.Canny(gray, 10, 250)
cv2.imwrite("edged.jpg", edged)
# закрываем контуры чтобы не осталось промежутков между пикселями(пока не знаю зачем...)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed.jpg", closed)

# ищем контуры, проверим, сколько вообще определилось посторонних (не фон) объектов.
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
total = 0

# т.к тетрадь, линейка - прямоугольник, хотя бы ищем контур с четырьмя вершинами
# цикл по контурам
for c in cnts:
    # аппроксимируем (сглаживаем) контур
    peri = cv2.arcLength(c, True) # находим периметр
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # аппроксируем контур, т.к из-за теней и зашумления вероятность нахождения
    # ровно четырех вершин не особо велика

    # если у контура 4 вершины, предполагаем, что это наш объект
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4) # обрисовываем объект, чтобы понять верно или нет
        total += 1
print('Найдено объектов:', total)
cv2.imwrite("output3.jpg", image)