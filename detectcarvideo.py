import cv2
import numpy as np
from time import sleep

largura_min = 80
altura_min = 80

offset = 6
pos_linha = 550

delay = 60

detect = []
carros = 0

def pega_centro(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo)           # video slowly or not
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)    # grey scale or color
    blur=cv2.GaussianBlur(grey, (3,3), 5)    # blur with grey color
    img_sub= subtracao.apply(blur)          # blur with grey color
    dilat =cv2.dilate(img_sub, np.ones((5, 5)))   # dilating-> moving body or transport in to white color

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)      #closing small holes inside the foreground
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE ,kernel)          #closing small holes inside the foreground
    contorno,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)         #find the contours

    cv2.line(frame1, (0, pos_linha), (1300, pos_linha), (255, 127, 0), 3)   # draw the line on the picture


    for(i,c) in enumerate(contorno):     #adds a counter to an iterable or on array
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w>=largura_min) and (h>= altura_min)
        if not validar_contorno:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0),2)       #Drawing a rectangle
        centro = pega_centro(x,y,w,h)
        detect.append(centro)
        # cv2.circle(frame1, centro, 4, (0, 0, 255), -1)       # drawing small red circle on finding object which is car
        for (x,y) in detect:
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1
                # cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 255), 3)
                detect.remove((x,y))
                print("Car is detected : " + str(carros))
    cv2.putText(frame1, "VEHICLE COUNT: "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video original", frame1)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()

