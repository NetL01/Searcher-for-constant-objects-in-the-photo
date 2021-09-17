import datetime
import cv2

colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))# Определить цвет прямоугольника

cap = cv2.VideoCapture("2.avi") # Параметр 0, чтобы открыть камеру, имя файла, чтобы открыть видео

fgbg = cv2.createBackgroundSubtractorMOG2()# Гибридный алгоритм гауссовского фонового моделирования

fourcc = cv2.VideoWriter_fourcc(*'XVID')# Установить сохранить формат изображения
out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 10.0, (768,576))# Разрешение должно соответствовать оригинальному видео


while True:
    ret, frame = cap.read()  # Читать картинку
    fgmask = fgbg.apply(frame)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Морфологический шум
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # Открыть операцию шумоподавления

    _ ,contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Найти перспективы

    count=0
    for cont in contours:
        Area = cv2.contourArea(cont)  # Рассчитать площадь контура
        if Area < 300:  # Форма с площадью фильтра менее 10
            continue

        count += 1  # Количество плюс один

        print("{}-prospect:{}".format(count,Area),end="  ") # Распечатать область каждого переднего плана

        rect = cv2.boundingRect(cont) # Извлечь прямоугольные координаты

        print("x:{} y:{}".format(rect[0],rect[1]))#Печать координат

        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),colour[count%6],1)# Нарисуйте прямоугольник на исходном изображении
        cv2.rectangle(fgmask,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0xff, 0xff, 0xff), 1)  # Рисуем прямоугольник на черном и белом переднем плане

        y = 10 if rect[1] < 10 else rect[1]  # Предотвратить нумерацию за пределы картинки
        cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # Напишите число на переднем плане



    cv2.putText(frame, "count:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) # Показать всего
    cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    print("----------------------------")

    cv2.imshow('frame', frame)# Отметить исходное изображение
    cv2.imshow('frame2', fgmask)  # Отображение переднего плана и фона в черно-белом
    out.write(frame)
    k = cv2.waitKey(30)&0xff  # Нажмите Esc для выхода
    if k == 27:
        break


out.release()# Выпустить файл
cap.release()
cv2.destoryAllWindows()# Закрыть все окна