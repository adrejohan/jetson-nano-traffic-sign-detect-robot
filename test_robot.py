from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import threading
from random import randint
#from jetbot import Robot

################ Initialization ################
detections = []
netMain = None
metaMain = None
altNames = None
Quit = True

#robot=Robot()

print("Choose Weather :")
print("1. Normal")
print("2. Drizzle")
print("3. Heavy Rain")
print("4. Storm")
print("5. Fog")

eff = input('Input Weather : ')
if eff == 2 :
    print("Weather Choosen : Drizzle")
    
elif eff == 3 :
    print("Weather Choosen : Heavy Rain")

elif eff == 4 :
    print("Weather Choosen : Storm")

elif eff == 5 :
    print("Weather Choosen : Fog")

else :
    print("Weather Choosen : Normal")

def Control () :
    def kiri () :
        #robot.forward(speed=0.6)
        print("Robot Lurus")
        time.sleep(1.6)
        #robot.stop()
        #robot.set_motors(-0.2,0.99)
        print("Robot Kiri")
        time.sleep(1)
        #robot.stop()
        #input("Press Enter to continue...")
        
    def kanan () :
        #robot.forward(speed=0.6)
        print("Robot Lurus")
        time.sleep(2.2)
        #robot.stop()
        #robot.set_motors(0.99,-0.2)
        print("Robot Kanan")
        time.sleep(1)
        #robot.forward(speed=0.6)
        print("Robot Lurus")
        time.sleep(0.5)
        #robot.stop()
        #input("Press Enter to continue...")

    while not Quit :
        if detections != [] :
            if detections[0][0].decode() == 'berhenti' :
                #robot.forward(speed=0.6)
                print("Robot Lurus")
                time.sleep(1)
                #robot.stop()
                print("Robot Berhenti")
                #input("Press Enter to continue...")

            elif detections[0][0].decode() == 'belok-kiri' :
                kiri()
                
            elif detections[0][0].decode() == 'arah-kiri' :
                if len(detections) == 2 :
                    if detections[1][0].decode() == 'arah-kanan' :
                        flag = randint(0, 1)
                        if flag == 0 :
                            kiri()
                        else :
                            kanan()
                else :
                    kiri()

            elif detections[0][0].decode() == 'arah-kanan' :
                if len(detections) == 2 :
                    if detections[1][0].decode() == 'arah-kiri' :
                        flag = randint(0, 1)
                        if flag == 0 :
                            kanan()
                        else :
                            kiri()
                else :
                    kanan()

            else :
                kanan()

        else :
            #robot.forward(speed=0.6)
            print("Robot Lurus")

def Image () :
    ###################### HLS #############################
    def hls(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_HLS=[]
            image_list=image
            for img in image_list:
                eval('image_HLS.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HLS))')
        else:
            image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
        return image_HLS

    def hue(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_Hue=[]
            image_list=image
            for img in image_list:
                image_Hue.append(hls(img,src)[:,:,0])
        else:
            image_Hue= hls(image,src)[:,:,0]
        return image_Hue

    def lightness(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_lightness=[]
            image_list=image
            for img in image_list:
                image_lightness.append(hls(img,src)[:,:,1])
        else:
            image_lightness= hls(image,src)[:,:,1]
        return image_lightness

    def saturation(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_saturation=[]
            image_list=image
            for img in image_list:
                image_saturation.append(hls(img,src)[:,:,2])
        else:
            image_saturation= hls(image,src)[:,:,2]
        return image_saturation

    ###################### HSV #############################
    def hsv(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_HSV=[]
            image_list=image
            for img in image_list:
                eval('image_HSV.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HSV))')
        else:
            image_HSV = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HSV)')
        return image_HSV

    def value(image,src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_value=[]
            image_list=image
            for img in image_list:
                image_value.append(hsv(img,src)[:,:,2])
        else:
            image_value= hsv(image,src)[:,:,2]
        return image_value

    ###################### BGR #############################
    def bgr(image, src='RGB'):
        verify_image(image)
        if(is_list(image)):
            image_BGR=[]
            image_list=image
            for img in image_list:
                eval('image_BGR.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2BGR))')
        else:
            image_BGR= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2BGR)')
        return image_BGR

    ###################### RGB #############################
    def rgb(image, src='BGR'):
        verify_image(image)
        if(is_list(image)):
            image_RGB=[]
            image_list=image
            for img in image_list:
                eval('image_RGB.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB))')
        else:
            image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
        return image_RGB

    def red(image,src='BGR'):
        verify_image(image)
        if(is_list(image)):
            image_red=[]
            image_list=image
            for img in image_list:
                i= eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
                image_red.append(i[:,:,0])
        else:
            image_red= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,0]')
        return image_red

    def green(image,src='BGR'):
        verify_image(image)
        if(is_list(image)):
            image_green=[]
            image_list=image
            for img in image_list:
                i= eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
                image_green.append(i[:,:,1])
        else:
            image_green= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,1]')
        return image_green

    def blue(image,src='BGR'):
        verify_image(image)
        if(is_list(image)):
            image_blue=[]
            image_list=image
            for img in image_list:
                i=eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
                image_blue.append(i[:,:,2])
        else:
            image_blue= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,2]')
        return image_blue


    err_not_np_img= "not a numpy array or list of numpy array"
    err_img_arr_empty="Image array is empty"
    err_row_zero="No. of rows can't be <=0"
    err_column_zero="No. of columns can't be <=0"
    err_invalid_size="Not a valid size tuple (x,y)"
    err_caption_array_count="Caption array length doesn't matches the image array length"

    def is_numpy_array(x):
        return isinstance(x, np.ndarray)
    def is_tuple(x):
        return type(x) is tuple
    def is_list(x):
        return type(x) is list
    def is_numeric(x):
        return type(x) is int
    def is_numeric_list_or_tuple(x):
        for i in x:
            if not is_numeric(i):
                return False
        return True


    def verify_image(image):
        if is_numpy_array(image):
            pass
        elif(is_list(image)):
            image_list=image
            for img in image_list:
                if not is_numpy_array(img):
                    raise Exception(err_not_np_img)
        else:
            raise Exception(err_not_np_img)

    err_rain_slant="Numeric value between -20 and 20 is allowed"
    err_rain_width="Width value between 1 and 5 is allowed"
    err_rain_length="Length value between 0 and 100 is allowed"
    def generate_random_lines(imshape,slant,drop_length,rain_type):
        drops=[]
        area=imshape[0]*imshape[1]
        no_of_drops=area//600

        if rain_type.lower()=='drizzle':
            no_of_drops=area//770
            drop_length=10
        elif rain_type.lower()=='heavy':
            drop_length=30
        elif rain_type.lower()=='torrential':
            no_of_drops=area//500
            drop_length=60

        for i in range(no_of_drops): ## If You want heavy rain, try increasing this
            if slant<0:
                x= np.random.randint(slant,imshape[1])
            else:
                x= np.random.randint(0,imshape[1]-slant)
            y= np.random.randint(0,imshape[0]-drop_length)
            drops.append((x,y))
        return drops,drop_length

    def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops):
        imshape = image.shape
        image_t= image.copy()
        for rain_drop in rain_drops:
            cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
        image= cv2.blur(image_t,(7,7)) ## rainy view are blurry
        brightness_coefficient = 0.7 ## rainy days are usually shady
        image_HLS = hls(image) ## Conversion to HLS
        image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
        image_RGB= rgb(image_HLS,'hls') ## Conversion to RGB
        return image_RGB

    ##rain_type='drizzle','heavy','torrential'
    def add_rain(image,slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='None'): ## (200,200,200) a shade of gray
        verify_image(image)
        slant_extreme=slant
        if not(is_numeric(slant_extreme) and (slant_extreme>=-20 and slant_extreme<=20)or slant_extreme==-1):
            raise Exception(err_rain_slant)
        if not(is_numeric(drop_width) and drop_width>=1 and drop_width<=5):
            raise Exception(err_rain_width)
        if not(is_numeric(drop_length) and drop_length>=0 and drop_length<=100):
            raise Exception(err_rain_length)

        if(is_list(image)):
            image_RGB=[]
            image_list=image
            imshape = image[0].shape
            if slant_extreme==-1:
                slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
            rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
            for img in image_list:
                output= rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
                image_RGB.append(output)
        else:
            imshape = image.shape
            if slant_extreme==-1:
                slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
            rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
            output= rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
            image_RGB=output

        return image_RGB

    err_fog_coeff="Fog coeff can only be between 0 and 1"
    def add_blur(image, x,y,hw,fog_coeff):
        overlay= image.copy()
        output= image.copy()
        alpha= 0.08*fog_coeff
        rad= hw//2
        point=(x+hw//2, y+hw//2)
        cv2.circle(overlay,point, int(rad), (255,255,255), -1)
        cv2.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)
        return output

    def generate_random_blur_coordinates(imshape,hw):
        blur_points=[]
        midx= imshape[1]//2-2*hw
        midy= imshape[0]//2-hw
        index=1
        while(midx>-hw or midy>-hw):
            for i in range(hw//10*index):
                x= np.random.randint(midx,imshape[1]-midx-hw)
                y= np.random.randint(midy,imshape[0]-midy-hw)
                blur_points.append((x,y))
            midx-=3*hw*imshape[1]//sum(imshape)
            midy-=3*hw*imshape[0]//sum(imshape)
            index+=1
        return blur_points

    def add_fog(image, fog_coeff=-1):
        verify_image(image)

        if(fog_coeff!=-1):
            if(fog_coeff<0.0 or fog_coeff>1.0):
                raise Exception(err_fog_coeff)
        if(is_list(image)):
            image_RGB=[]
            image_list=image
            imshape = image[0].shape

            for img in image_list:
                if fog_coeff==-1:
                    fog_coeff_t=random.uniform(0.3,1)
                else:
                    fog_coeff_t=fog_coeff
                hw=int(imshape[1]//3*fog_coeff_t)
                #haze_list= generate_random_blur_coordinates(imshape,hw)
                #for haze_points in haze_list:
                    #img= add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
                img = cv2.blur(img ,(hw//10,hw//10))
                image_RGB.append(img)
        else:
            imshape = image.shape
            if fog_coeff==-1:
                fog_coeff_t=random.uniform(0.3,1)
            else:
                fog_coeff_t=fog_coeff
            hw=int(imshape[1]//3*fog_coeff_t)
            #haze_list= generate_random_blur_coordinates(imshape,hw)
            #for haze_points in haze_list:
                #image= add_blur(image, haze_points[0],haze_points[1], hw,fog_coeff_t)
            image = cv2.blur(image ,(hw//10,hw//10))
            image_RGB = image

        return image_RGB

    def convertBack(x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def cvDrawBoxes(detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img

    global metaMain, netMain, altNames, detections, Quit
    configPath = "./cfg/yolov3-tiny-traffic-tes.cfg"
    weightPath = "./yolov3-tiny-traffic_last.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE , -5)
    cap.set(3, 640)
    cap.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (288, 288))

    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    Control.start()
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()

        if eff == 2 :
            weather = add_rain(frame_read, rain_type='drizzle')
    
        elif eff == 3 :
            weather = add_rain(frame_read, rain_type='heavy', slant=10)

        elif eff == 4 :
            weather = add_rain(frame_read, rain_type='torrential', slant=20)

        elif eff == 5 :
            weather = add_fog(frame_read, fog_coeff=0.7)

        else :
            weather = frame_read
     
        frame_rgb = cv2.cvtColor(weather, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.7)
        Quit = False
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(detections[1])
        cv2.imshow('Demo',image)
        out.write(image)
        c = cv2.waitKey(1)

        if c & 0xFF == ord('q'):
            Quit = True
            break

    cap.release()
    out.release()
    print (threading.currentThread().getName(), 'Shutting Down')
    print (threading.currentThread().getName(), 'All Done')

Control = threading.Thread(name = 'Control', target = Control, args = ())
Image = threading.Thread(name = 'Image', target = Image,args = ())

Image.start()
