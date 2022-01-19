import numpy as np
import cv2 as cv

cv.namedWindow('Controls')

UP = 1
DOWN = 0
HIDDEN = 0
MAX_HISTORY = 5

cap = cv.VideoCapture(1)



y_coord = 0
direction = UP
bounces = 0
state = UP
history = []
avg_history = 0

interface = {}
interface[DOWN] = "down"
interface[UP] = "up"

def Average(lst):
    return sum(lst) / len(lst)

def nothing(x):
    pass

def resetBounces():
    global bounces
    bounces = 0

cv.createTrackbar('H_lower', 'Controls', 0, 179, nothing)
cv.createTrackbar('H_upper', 'Controls', 40, 179, nothing)

cv.createTrackbar('S_lower', 'Controls', 164, 255, nothing)
cv.createTrackbar('S_upper', 'Controls', 255, 255, nothing)

cv.createTrackbar('V_lower', 'Controls', 164, 255, nothing)
cv.createTrackbar('V_upper', 'Controls', 255, 255, nothing)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    lower_orange = np.array([0,164,164])
    upper_orange = np.array([40,255,255])

    # get current positions of trackbars
    lower_orange[0] = cv.getTrackbarPos('H_lower','Controls')
    lower_orange[1] = cv.getTrackbarPos('S_lower','Controls')
    lower_orange[2] = cv.getTrackbarPos('V_lower','Controls')

    upper_orange[0] = cv.getTrackbarPos('H_upper','Controls')
    upper_orange[1] = cv.getTrackbarPos('S_upper','Controls')
    upper_orange[2] = cv.getTrackbarPos('V_upper','Controls')


    blur = cv.GaussianBlur(frame,(5,5),0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_orange, upper_orange)
    mask = cv.GaussianBlur(mask,(5,5),0)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask = mask)

    # Finding the largest cluster of orange pixels
    # using findContours func to find the none-zero pieces
    contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        y_coord = y + h/2
    
    history.append(y_coord)
    if len(history) > MAX_HISTORY:
        history.pop(0)

    if Average(history) > avg_history:
        direction = DOWN
    else:
        direction = UP

    avg_history = Average(history)

    if direction == UP and state == DOWN:
        bounces += 1
    state = direction
    
    frame = cv.rectangle(frame, (0,0), (160,90), (20,20,20), -1)

    cv.putText(
        frame, #numpy array on which text is written
        f"y-coord: {y_coord}", #text
        (0,20), #position at which writing has to start
        cv.FONT_HERSHEY_SIMPLEX, #font family
        0.4, #font size
        (0, 0, 255), #font color
        1) #font stroke

    cv.putText(
        frame, #numpy array on which text is written
        f"direction: {interface[state]}", #text
        (0,40), #position at which writing has to start
        cv.FONT_HERSHEY_SIMPLEX, #font family
        0.4, #font size
        (0, 0, 255), #font color
        1) #font stroke

    cv.putText(
        frame, #numpy array on which text is written
        f"count: {bounces}", #text
        (0,60), #position at which writing has to start
        cv.FONT_HERSHEY_SIMPLEX, #font family
        0.4, #font size
        (0, 0, 255), #font color
        1) #font stroke

    cv.putText(
        frame, #numpy array on which text is written
        f"H: {lower_orange[0]}, S: {lower_orange[1]}, V: {lower_orange[2]}", #text
        (0,80), #position at which writing has to start
        cv.FONT_HERSHEY_SIMPLEX, #font family
        0.4, #font size
        (0, 0, 255), #font color
        1) #font stroke

    # show the shit
    #out.write(frame)
    cv.imshow('frame', frame)
    
    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('r'):
        bounces = 0
    
    
# Release everything if job is finished
cap.release()
#out.release()
cv.destroyAllWindows()

