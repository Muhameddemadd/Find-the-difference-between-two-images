import cv2
import numpy as np
from skimage import io
from skimage.morphology import remove_small_objects

img = cv2.imread('pic1.jpg')
img2= cv2.imread('pic2.jpg')

def set_shapes(pic1,pic2):
    global width
    global height
    pic1 = cv2.cvtColor(io.imread('pic1.jpg'), cv2.COLOR_RGB2GRAY)
    pic2=cv2.cvtColor(io.imread('pic2.jpg'), cv2.COLOR_RGB2GRAY)
    pic1= cv2.threshold(pic1, 128, 255, cv2.THRESH_BINARY)[1]
    pic2 = cv2.threshold(pic2, 128, 255, cv2.THRESH_BINARY)[1]
    dimen1=pic1.shape
    dimen2=pic2.shape
    height1=pic1.shape[0]
    height2=pic2.shape[0]
    width1=pic1.shape[1]
    width2=pic2.shape[1]
    
    if height1>=height2:
        height=height1
    elif height1<=height2:
        height=height2
    if width1>=width2:
        width=width1
    elif width1<=width2:
        width=width2
    dsize=(width,height)   
    pic1=cv2.resize(pic1,dsize)
    pic2=cv2.resize(pic2,dsize)
    return pic1,pic2

def normalize(pic1):
    hsv = cv2.cvtColor(pic1, cv2.COLOR_RGB2HSV)
    hue=cv2.inRange(hsv,np.array([0,27,133]),np.array([180,255,255]))
    thresh=cv2.bitwise_and(hsv,hsv,mask=hue)
    thresh= cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)

    # morphologically close the gaps between purple and blue tubes
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    #cv2.imwrite('closing_result.png', thresh)
    #cv2.imshow('closing_result.png', thresh)

    # morphological opening with horizontal and vertical kernels
    h_kernel = np.zeros((11, 11), dtype=np.uint8)
    h_kernel[5, :] = 1
    #print(h_kernel)

    v_kernel = np.zeros((11, 11), dtype=np.uint8)
    v_kernel[:,5] = 1

    h_tubes = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=6)
    v_tubes = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=7)

    cv2.imwrite('horizontal_tubes.png', h_tubes)
    cv2.imshow('horizontal_tubes.png', h_tubes)

    cv2.imwrite('vertical_tubes.png', v_tubes)
    cv2.imshow('vertical_tubes.png', v_tubes)

    # find contours and draw rectangles with constant widths through centers
    h_contours = cv2.findContours(h_tubes, cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
    h_lines = np.zeros(thresh.shape, np.uint8)

    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y += int(np.floor(h / 2) - 4)
        cv2.rectangle(h_lines, (x, y), (x + w, y + 8), 255, -1)

    v_contours = cv2.findContours(v_tubes, cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
    v_lines = np.zeros(thresh.shape, np.uint8)

    for cnt in v_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x += int(np.floor(w / 2) - 4)
        cv2.rectangle(v_lines, (x, y), (x + 8, y + h), 255, -1)

    # combine horizontal and vertical lines
    all_lines = h_lines | v_lines

   # cv2.imwrite('all_lines.png', all_lines)
    #cv2.imshow('all_lines.png', all_lines)


    # remove small objects around the intersections
    xor = np.bool8(h_lines ^ v_lines)
    removed = xor ^ remove_small_objects(xor, 350)

    result = all_lines & ~removed * 255
    result = np.asarray(result, dtype=np.uint8)
    cv2.imshow("all_img",result)
    return result

    
def horizontal_detec(pic1,pic2):
    global mask11
    global mask22
    
#detection horizontal contour
    horizontal_size1 = 11
    horizontalStructure1 = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size1, 1))
    horizontal_size2 = 11
    horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size2, 1))
#use morphology operations
    mask1 = cv2.morphologyEx(pic1, cv2.MORPH_OPEN, horizontalStructure1)
    mask2 = cv2.morphologyEx(pic2, cv2.MORPH_OPEN, horizontalStructure2)
    mask11 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((7, 8), np.uint8))
    mask22 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((7 ,8), np.uint8))
#find vectors parameter : (x,y), and horizontal length
    h_contours = cv2.findContours(mask11, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    arr=[]
    for cnt in h_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        arr.append([x,y,w])
    h_contours2 = cv2.findContours(mask22, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    arr2=[]
    arr3=[]
    for cnt2 in h_contours2:
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        arr2.append([x2,y2,w2])
#equalize to shapes        
    if len(arr)>len(arr2):
        arr2.append([0,0,0])
    elif len(arr)<len(arr2):
        arr.append([0,0,0])
    #print(arr)
    #print(arr2)
                
#searching algrothim        
    i=1      
    for element in arr2:
        
            x=element[0]
            y=element[1]
            w=element[2]
            print("x1",x)
            for element2 in arr: 
                 x2=element2[0]
                 y2=element2[1]
                 w2=element2[2]
                 if abs(x-x2)<10 and abs(y-y2)<10 and abs(w-w2)<10:
                         break  
                 elif i == len(arr2):
                     if x-x2>0 or y-y2>0 or w-w2>0:
                           arr3=[x,y,w]
                     elif x2-x>0 or y2-y>0 or w2-w>0:
                           arr3=[x2,y2,w2]                                             
            i=i+1
#parameter of differnce            
    x3=arr3[0]
    y3=arr3[1]
    w3=arr3[2]
    print(x3, y3,w3)
    mask22final=cv2.cvtColor(mask22,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(mask22final,(x3,y3),(x3+w3,y3+10),(255,0,0),5)
    cv2.imshow('img1',mask22final)
    cv2.imwrite("Horizontal.jpg", mask22final)

    return arr3 

def vertical_detection(pic1,pic2):
    global mask11_v
    global mask22_v
    vertical_size1 = 20
    verticalStructure1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size1))

    vertical_size2 = 20
    verticalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size2))

    mask1_v = cv2.morphologyEx(pic1, cv2.MORPH_OPEN, verticalStructure1)
    mask2_v = cv2.morphologyEx(pic2, cv2.MORPH_OPEN, verticalStructure2)

    mask11_v = cv2.morphologyEx(mask1_v, cv2.MORPH_CLOSE, np.ones((7, 8), np.uint8))
    mask22_v = cv2.morphologyEx(mask2_v, cv2.MORPH_CLOSE, np.ones((7 ,8), np.uint8))

    #mask11_v = cv2.dilate(mask11_v, verticalStructure1,iterations=3)
    #mask22_v = cv2.dilate(mask22_v, verticalStructure2,iterations=3)

    v_contours = cv2.findContours(mask11_v, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.imshow('img45',mask11_v)
    arr_v=[]
    print(len(v_contours))
    for cnt_v in v_contours:
          
            x_v, y_v, w_v, h_v = cv2.boundingRect(cnt_v)
            arr_v.append([x_v,y_v,h_v])
            print(arr_v)
    
    v_contours2 = cv2.findContours(mask22_v, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    arr_v2=[]
    arr_v3=[]
    for cnt_v2 in v_contours2:
        
        x_v2, y_v2, w_v2, h_v2 = cv2.boundingRect(cnt_v2)
        arr_v2.append([x_v2,y_v2,h_v2])
    if len(arr_v)>len(arr_v2):
        arr_v2.append([0,0,0])
    elif len(arr_v)<len(arr_v2):
        arr_v.append([0,0,0])
    #print(arr_v)
    #print(arr_v2)
                    
    i_v=1      
    for element in arr_v2:
            
                x_v=element[0]
                y_v=element[1]
                h_v=element[2]
                #print("x1",xv)
                for element2 in arr_v: 
                     x_v2=element2[0]
                     y_v2=element2[1]
                     h_v2=element2[2]
                     if abs(x_v-x_v2)<10 and abs(y_v-y_v2)<10 and abs(h_v-h_v2)<10:
                             break  
                     elif i_v == len(arr_v2):
                         if x_v-x_v2>0 or y_v-y_v2>0 or h_v-h_v2>0:
                               arr_v3=[x_v,y_v,h_v]
                         elif x_v2-x_v>0 or y_v2-y_v>0 or h_v2-h_v>0:
                               arr_v3=[x_v2,y_v2,h_v2]
                                             
                i_v=i_v+1
                
    x_v3=arr_v3[0]
    y_v3=arr_v3[1]
    h_v3=arr_v3[2]
    print(x_v3, y_v3,h_v3)
    mask22_vfinal=cv2.cvtColor(mask22_v,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(mask22_vfinal,(x_v3,y_v3),(x_v3+10,y_v3+h_v3),(255,0,0),5)
    cv2.imshow('img2',mask22_vfinal)
    cv2.imwrite("Vertical.jpg", mask22_vfinal)

    return arr_v3

def all_lines_detection(list1,list2):#(horizontal,vertical)
    diffence_x=0
    diffence_y=0
    if abs(list1[0]-list2[0])<10:
        if (list1[0]-list2[0])>=0:
            diffence_x=list1[0]
        elif (list2[0]-list2[0])>0:
            diffence_x=list2[0]
        if (list1[1]-list2[1])>0:
            diffence_y=list1[1]
        elif (list2[1]-list2[1])>0:
            diffence_y=list2[1]   
    all_lines1= mask11_v | mask11      
    all_lines2= mask22_v | mask22      
    all_lines1=cv2.cvtColor(all_lines1,cv2.COLOR_GRAY2RGB)
    all_lines2=cv2.cvtColor(all_lines2,cv2.COLOR_GRAY2RGB)
    width=diffence_x+list1[2]+10
    heigth=diffence_y-(list1[1]-list2[1])
    #cv2.rectangle( img,(diffence_x-20, diffence_y+20),(width,heigth),(255,0,0),5)
    cv2.rectangle( img2,(diffence_x-20, diffence_y+20),(width,heigth),(255,0,0),5) 
    cv2.imshow("output1",  all_lines1)
    cv2.imshow("output2",  all_lines2)
    cv2.imwrite("output1.jpg",  all_lines1)
    cv2.imwrite("output2.jpg",  all_lines2)
   
result2=normalize(img2)
result1=normalize(img)
#result1,result2=set_shapes(result1,result2)
horizotnal_val=horizontal_detec(result1,result2)
vertical_val=vertical_detection(result1,result2)
all_lines_detection(horizotnal_val,vertical_val)    
cv2.imshow('img3',img2)
cv2.imshow('img4',img)
cv2.waitKey(1000000)

