import win32gui, win32api, win32con
import cv2
from PIL import ImageGrab
import numpy as np
from datetime import datetime
import time



def mouse_click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def img_to_str(image,area,alfa,min_countour_area,max_countour_area,black=False):
    image = image[area[0][1]:area[1][1],area[0][0]:area[1][0]]
    out = np.zeros(image.shape,np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if black == False:
        i = 0
        while len(image) > i:
            j = 0
            while len(image[i]) > j:
                if image[i][j] < alfa:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
                j += 1
            i += 1
    elif black == True:
        i = 0
        while len(image) > i:
            j = 0
            while len(image[i]) > j:
                if image[i][j] >= alfa:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
                j += 1
            i += 1   
    img_ret = image
    contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    new_dice_cost_int = ''
    for cnt in contours:
        if cv2.contourArea(cnt) > min_countour_area and cv2.contourArea(cnt) < max_countour_area:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h>10:
                
                #cv2.rectangle(norm,(x,y),(x+w,y+h),(0,0,255),2)
                roi = image[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                string = str(int((results[0][0])))
                new_dice_cost_int += string
                cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
    new_dice_cost_int2 = -1
    if new_dice_cost_int != '':
        new_dice_cost_int2 = ''
        for i in new_dice_cost_int:
            new_dice_cost_int2 = i + new_dice_cost_int2
    return new_dice_cost_int2
        

def spawn(image):
    value = img_to_str(image,MP_value,172,10,100)
    cost = img_to_str(image,new_dice_cost,225,10,100)
    if int(value) >= int(cost) and int(value) != -1:
        mouse_click(x0 + new_dice_button[0],y0 + new_dice_button[1])
        return True
    else:
        return False

def cord_union_dice(game_board,image,empty_const):
    for i in game_board:
        temp_image = image[i[0][1]-rect:i[0][1]+rect,i[0][0]-rect:i[0][0]+rect]
        res = cv2.matchTemplate(temp_image,empty_const,cv2.TM_CCOEFF_NORMED)
        if res <= 0.5:
            i[1] = temp_image
        else:
            i[1] = []
    ii = 0
    for i in game_board:
        if i[1] != []:
            jj = 0
            for j in game_board:
                temp_image = image[j[0][1]-rect:j[0][1]+rect,j[0][0]-rect:j[0][0]+rect]
                res = cv2.matchTemplate(temp_image,i[1],cv2.TM_CCOEFF_NORMED)
                if res >= 0.93:
                    if ii != jj:
                        mass = [i[0],j[0]]
                        return mass
                jj += 1       
        ii += 1
        
def union_dice(mass):
    if mass != None:
        time.sleep(time_sleep)
        win32api.SetCursorPos((x0 + mass[0][0],y0 + mass[0][1]))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x0 + mass[0][0],y0 + mass[0][1],0,0)
        time.sleep(time_sleep)
        win32api.SetCursorPos((x0 + mass[1][0],mass[1][1]))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x0 + mass[1][0],y0 + mass[1][1],0,0)
        time.sleep(time_sleep)
        
def upgrade_improv(image,area):
    area_price = [(area[0] - temp_rect_improve[0],area[1] - temp_rect_improve[1]),(area[0] + temp_rect_improve[0],area[1] + temp_rect_improve[1])]
    improve_price = img_to_str(image,area_price,170,10,100,True)
    value = img_to_str(image,MP_value,172,10,100)
    if int(value) >= int(improve_price) and int(improve_price) != 0 and int(improve_price) != -1:
        mouse_click(x0 + area[0],y0 + area[1])
        return True
    else:
        return False
        
    
    

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)





while False:
    hwnd = win32gui.FindWindow(None, 'NoxPlayer')
    x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
    screen = np.asarray(ImageGrab. grab())
    image = screen[y0:y1, x0:x1]
    #image = cv2.imread('1.png')
    h = y1 - y0
    w = x1 - x0
    time_sleep = 0.2
    game_board = [[[round(w/3.7),round(h/1.677)],[]],
                  [[round(w/2.6),round(h/1.677)],[]],
                  [[round(w/2),round(h/1.677)],[]],
                  [[round(w/1.625),round(h/1.677)],[]],
                  [[round(w/1.37),round(h/1.677)],[]],
                  [[round(w/3.7),round(h/1.523)],[]],
                  [[round(w/2.6),round(h/1.523)],[]],
                  [[round(w/2),round(h/1.523)],[]],
                  [[round(w/1.625),round(h/1.523)],[]],
                  [[round(w/1.37),round(h/1.523)],[]],
                  [[round(w/3.7),round(h/1.394)],[]],
                  [[round(w/2.6),round(h/1.394)],[]],
                  [[round(w/2),round(h/1.394)],[]],
                  [[round(w/1.625),round(h/1.394)],[]],
                  [[round(w/1.37),round(h/1.394)],[]]]
    
    area_improv = [[round(w/5.15),round(h/1.052)],
                   [round(w/2.82),round(h/1.052)],
                   [round(w/1.94),round(h/1.052)],
                   [round(w/1.475),round(h/1.052)],
                   [round(w/1.193),round(h/1.052)]]
    
    rect = round(h/34)
    
    empty_const = cv2.imread('empty.png')
    
    temp_rect_improve =  [round(h/70),round(w/55)]

    new_dice_button = (round(w/2),round(5*h/6.05))
    new_dice_cost = [(round((w/2)-(h/140)),round(239*h/280-w/50)),(round((w/2)+(h/40)),round(239*h/280+w/50))]
    MP_value = [(round((w/2)-(h/6)),round(239*h/279-w/50)),(round((w/2)-(h/9.4)),round(239*h/280+w/50))]
    

    ##############
    Main_glob_bool = False
    Main_glob_bool = spawn(image)
    time.sleep(time_sleep)
    mass = cord_union_dice(game_board,image,empty_const)
    union_dice(mass)
    time.sleep(time_sleep)
    for area in area_improv:
        if Main_glob_bool == False:
            Main_glob_bool = upgrade_improv(image,area)
            time.sleep(time_sleep)
    time.sleep(time_sleep)
    
    ##############
    # y = round(h/1.052)
    # x0 = round(w/5.15)  x1= round(w/2.82) x2 = round(w/1.94) x3= round(w/1.475)  x4= round(w/1.193)
    #temp_x = round(w/2.82) 
    #temp_y = round(h/1.052)
    #temp_rect_improve = 20
    #cv2.rectangle(image,(temp_x,temp_y),(temp_x,temp_y),(0,0,255),1)
    
    #image = image[area_improv[0][1] - temp_rect_improve[1]:area_improv[0][1] + temp_rect_improve[1], area_improv[0][0] - temp_rect_improve[0]:area_improv[0][0] + temp_rect_improve[0]]
    #test = [(area_improv[0][0] - temp_rect_improve[0],area_improv[0][1] - temp_rect_improve[1]),(area_improv[0][0] + temp_rect_improve[0],area_improv[0][1] + temp_rect_improve[1])]
    
    
    
    
   
    #cv2.imwrite("empty.png", )  
    #cv2.imshow("Image", image)
    
    #key = cv2.waitKey(0)

    #time.sleep(0.5)
    #cv2.destroyAllWindows()
    #timestamp = datetime.today().timestamp()
    #cv2.imwrite("constanta.png", image)
    #cv2.imwrite("set_training/value" + str(timestamp) + ".png", image[MP_value[0][1]:MP_value[1][1],MP_value[0][0]:MP_value[1][0]])
   



