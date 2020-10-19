import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import itertools
import sounddevice as sd
import soundfile as sf
import threading
from os import path
import requests
import json
import imutils
import math
import sys
import os
from help_functions import *
from colors import bcolors
from itertools import groupby, combinations
from shapely import geometry
import matplotlib.pyplot as plt

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

#Ignoring small people detection
ignore_value = 200

#Ratio to calculate voilation
inside_main_zone = 0.8
boundary_zones_with_main_zone = 0.7
boundary_zones = 0.6

#Final Display Settings
width_input = 1280
full_screen = False
show_humans_detected = False

#Base for drawing red bounding box and playing warning.
social_distance_voilators = []
social_distance_voilators_box = []
timer_for_each_voilators = dict()
temp_timer_for_each_voilator = dict()

#From background calculation based on user inputs.
people_positions = []
nearest_distance_combo = []
sorted_points = []
zones = []
zone_with_distances = []
neighbour_zones_with_positions = []

#User input from GUI
video_path = ""
seconds = 0
audio_path = ""
audio_length = 0

#Extra variables needed to play warning.
time_to_wait = 0
waits = 5
data = None
fs = None

#Defining red color rgb value
COLOR_RED = (0, 0, 255)

######################################### 
# Load the background for the top-down view #
#########################################
print(bcolors.WARNING +"[ Loading Config file for settings] "+ bcolors.ENDC)

input_values = eval(open(r"./SupportingFiles/config.txt", 'r').read())
ignore_value = input_values["ignore_value"]
inside_main_zone = input_values['inside_main_zone']
boundary_zones_with_main_zone = input_values['boundary_zones_with_main_zone']
boundary_zones = input_values['boundary_zones']
width_input = input_values['width_input']
full_screen = input_values['full_screen']
show_humans_detected = input_values['show_humans_detected']
waits = input_values['gap']
COLOR_RED = input_values['color']


print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )
    
######################################### 
# Load the background for the top-down view #
#########################################
print(bcolors.WARNING +"[ Loading background calculation file for drawing grid] "+ bcolors.ENDC)

input_data = eval(open(r"F:\Gopal\Coding\20201009\SingleSystem\COVID-19_Social_Distance_Maintainer\SupportingFiles\background_data.txt", 'r').read())
nearest_distance_combo = input_data["nearest_distance_combo"]
sorted_points = input_data["all_points_sorted"]
zones = input_data["all_zones_tagged"]
zone_with_distances = input_data["zones_with_distances"]
neighbour_zones_with_positions = input_data["neighbour_zones_position_combo"]

print(bcolors.OKGREEN +" Done : [ Background calculation file loaded ] ..."+bcolors.ENDC )

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    input_size = FLAGS.size

    ######################################################
    # 				START THE VIDEO STREAM               #
    ######################################################

    if video_path == "0" or video_path.startswith("rtsp") or video_path.startswith("http") or video_path.startswith("https") or video_path.startswith("tcp") or video_path.startswith("udp"):
        try:
            good_to_write = True
            rtsp_cam(input_size, infer)
        except:
            good_to_write = False
            print("Exception occured. Something wrong 1.")
            
    else:	
        try:
            vs = cv2.VideoCapture(video_path)
            frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
            good_to_write = True
            work_with_video(vs, frame_per_seconds, input_size, infer)
        except:
            good_to_write = False
            print("Exception occured. Something wrong  or video finished.")



def rtsp_cam(input_size, infer):
    while True:
        vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
        if vs.isOpened():
            print('Camera is connected')
            #call function
            response = work_with_video(vs, frame_per_seconds, input_size, infer)
            if response == False:
                time.sleep(5)
                continue
        else:
            print('Camera not connected')
            vs.release()
            time.sleep(5)
            continue  



def work_with_video(video_s,fps,input_size, infer):
    global social_distance_voilators, social_distance_voilators_box, people_positions
    global timer_for_each_voilators, temp_timer_for_each_voilator

    output_video_1 = None
    start_timer = 0
    start_time_for_del = 0
    l_count = 0
    # Loop until the end of the video stream
    frame_to_check = int(fps/5)

    array_boxes_detected = []
    box_and_ground_points = []
    boxes_to_make_red = []
    array_centroids = []
    humans_detected = []

    while True:	
        if start_timer == 0:
            start_timer = time.time()
        if start_time_for_del == 0:
            start_time_for_del = time.time()
        if video_path == "0" or video_path.startswith("rtsp") or video_path.startswith("http") or video_path.startswith("https") or video_path.startswith("tcp") or video_path.startswith("udp"):
            # try:
            last_ret, latest_frame = video_s.read()
            if (last_ret is not None) and (latest_frame is not None):
                frame = latest_frame.copy()
                good_to_write = True 
            else:
                print("Camera is disconnected!")
                vs.release()
                return False
                good_to_write = False

        else:
            frame_exist, frame = video_s.read()
            good_to_write = True
        
        if l_count == 0 or l_count%frame_to_check==0:
            array_boxes_detected.clear()
            box_and_ground_points.clear()
            boxes_to_make_red.clear()
            array_centroids.clear()
            humans_detected.clear()
            social_distance_voilators.clear()
            social_distance_voilators_box.clear()
            people_positions.clear()

            # Resize the image to the correct size
            frame = imutils.resize(frame,width=width_input)
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # # To check time consumed by each frame calculation.
            # prev_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            final_boxes = boxes.numpy()
            final_scores = scores.numpy()
            final_classes = classes.numpy()

            array_boxes_detected_without_filter = get_human_box_detection(final_boxes,final_scores[0].tolist(),final_classes[0].tolist(),frame.shape[0],frame.shape[1])
            
            # # To check time consumed by each frame calculation.
            # curr_time = time.time()
            # exec_time = curr_time - prev_time
            # info = "time: %.2f ms" %(1000*exec_time)
            # print(info)

            for boxes in array_boxes_detected_without_filter:
                x1 = boxes[1]
                y1 = boxes[0]
                x2 = boxes[3]
                y2 = boxes[2]
                dist = int(calculateDistance(x1,y1,x2,y2))
                if dist>ignore_value:
                    array_boxes_detected.append(boxes)
            # Check if 2 or more people have been detected (otherwise no need to calculate)
            if len(array_boxes_detected)>=2:
                # Both of our lists that will contain the centroïds coordonates and the ground points
                array_centroids,humans_detected = get_centroids_and_groundpoints(array_boxes_detected)
                if show_humans_detected==True:
                    for hd in humans_detected:
                        cv2.circle(frame,tuple(hd),5,(0,255,0),-1)

                box_and_ground_points = list(zip(humans_detected,array_boxes_detected))

                zones_list = list(zones.values())
                people_at_various_zones = dict()
                for n,zone in enumerate(zones_list):
                    if len(zone)==4:
                        sorted_zone = zone[:2] + [zone[3]] + [zone[2]]
                        z = geometry.Polygon(sorted_zone)
                    else:
                        z = geometry.Polygon(zone)
                    # x,y = z.exterior.xy
                    # plt.plot(x,y)
                    # plt.show()
                    for hd in humans_detected:
                        human_point = geometry.Point(hd)
                        if f"Zone{n}" not in people_at_various_zones.keys():
                            people_at_various_zones[f"Zone{n}"] = []
                        if z.intersects(human_point):
                            people_at_various_zones[f"Zone{n}"].append(hd)
                            
                people_at_various_zones_list = list(people_at_various_zones.values())
                people_in_main_zone = people_at_various_zones_list[0]

                check_on_main_zone(people_in_main_zone,frame) 
                people_at_boundary_zones =  people_at_various_zones_list[1:]  

                filtered_people_at_boundary_zones = []  
                filtered_people_at_boundary_zones.append(people_at_boundary_zones[0]) 
                for n,people in enumerate(people_at_boundary_zones[1:]):
                    zone = []
                    if n==people_at_boundary_zones[-1]:
                        if p not in people_at_boundary_zones[0]:
                            zone.append(p)
                    for p in people:
                        p_at_b = filtered_people_at_boundary_zones[n]
                        if p not in p_at_b:
                            if p not in people_in_main_zone:
                                zone.append(p)
                    filtered_people_at_boundary_zones.append(zone)


                list_of_neighbour_positions = []
                for neighbour in neighbour_zones_with_positions:
                    if len(neighbour[-1])>1:
                        list_of_neighbour_positions.append(neighbour[-1])

                for n,people in enumerate(filtered_people_at_boundary_zones,1):
                    if people:
                        for p in people:
                            to_check_with = neighbour_zones_with_positions[n-1][-1]
                            if len(to_check_with)==1:
                                for v in people_positions:
                                    if v[0] == to_check_with[0]:
                                        a = p
                                        b = v[1]
                                        find_voilators(a,b,n,boundary_zones_with_main_zone)
                            else:
                                for pos in to_check_with:
                                    for v in people_positions:
                                        if v[0] == pos:
                                            find_voilators(p,v[1],n,boundary_zones_with_main_zone)
                        if len(people)>1:
                            comb = list(combinations(people,2))
                            for c in comb:
                                a = c[0]
                                b = c[1]
                                find_voilators(a,b,n,boundary_zones)

                        neighbour_zones = neighbour_zones_with_positions[n-1]
                        for neighbour in neighbour_zones[0][1]:
                            for k,zs in zones.items():
                                if sorted(zs) == sorted(neighbour):
                                    zone_id = int(''.join([i for i in k if i.isdigit()]))
                                    neighbour_people = filtered_people_at_boundary_zones[zone_id-1]
                                    if neighbour_people:
                                        for n1 in neighbour_people:
                                            for p1 in people:
                                                a = n1
                                                b = p1
                                                find_voilators(a,b,n,boundary_zones)


            for voilators in social_distance_voilators:
                for box_value in box_and_ground_points:
                    if voilators in box_value:
                        social_distance_voilators_box.append(box_value[1])
            

            #Drawing bounding box
            if len(social_distance_voilators_box)>=2:
                red_box(social_distance_voilators_box, frame)
            l_count += 1

        else:
            frame = imutils.resize(frame,width=width_input)
            if len(social_distance_voilators_box)>=2:
                red_box(social_distance_voilators_box, frame)
            if show_humans_detected==True:
                for hd in humans_detected:
                    cv2.circle(frame,tuple(hd),5,(0,255,0),-1)
            l_count += 1

        end_timer = time.time()
        current_time_value = end_timer-start_timer


        if current_time_value >= 1:
            if len(timer_for_each_voilators)==0:
                if len(social_distance_voilators)>=2:
                    for voil in social_distance_voilators:
                        timer_for_each_voilators[voil] = 1

            for voi in social_distance_voilators:
                result_to_confirm = []
                for key in timer_for_each_voilators:
                    if voi == key:
                        temp_dict = dict()
                        temp_dict[(key,voi)] = True
                        result_to_confirm.append(temp_dict)
                    else: 
                        x_minus = key[0]-10
                        x_plus = key[0]+10
                        y_minus = key[1]-10
                        y_plus = key[1]+10
                        key_minus = (x_minus, y_minus)
                        key_pm = (x_plus,y_minus)
                        key_plus = (x_plus, y_plus)
                        key_mp = (x_minus, y_plus)
                        area_to_check = [key_minus, key_pm, key_plus, key_mp]
                        voilators_point = geometry.Point(voi)
                        polygon_area = geometry.Polygon(area_to_check)

                        #Check if polygon is correctly plotted.
                        # xx,yy = polygon_area.exterior.xy
                        # plt.plot(xx,yy)
                        # plt.show()
    
                        if polygon_area.intersects(voilators_point):
                            temp_dict = dict()
                            temp_dict[(key,voi)] = True
                            result_to_confirm.append(temp_dict)
                        else:
                            temp_dict = dict()
                            temp_dict[(key,voi)] = False
                            result_to_confirm.append(temp_dict)
                val = []   
                for rtc in result_to_confirm:
                    val.append(list(rtc.values())[0])
                if any(val) == True:
                    for rtc in result_to_confirm:
                        if list(rtc.values())[0] == True:
                            keey = list(rtc.keys())[0][0]
                            timer_for_each_voilators[keey] += 1
                            break
                else:
                    for rtc in result_to_confirm:
                        keey = list(rtc.keys())[0][1]
                        timer_for_each_voilators[keey] = 1
                        break

            if len(timer_for_each_voilators)>1:
                t = timer_for_each_voilators.values()
                t_max = max(t)
                # print(t_max)
                # print(timer_for_each_voilators)
                # print()
                if t_max >= seconds and time_to_wait==0:
                    t_max = 0
                    threading.Thread(target = play_warning).start()
                    threading.Thread(target= waiting_time).start()
            start_timer = 0

        end_timer_for_del = time.time()
        check_to_delete = end_timer_for_del-start_time_for_del
        if check_to_delete>=2:
            if len(temp_timer_for_each_voilator) != 0:
                to_delete = []
                for element in timer_for_each_voilators:
                    if element in temp_timer_for_each_voilator:
                        ttfev = temp_timer_for_each_voilator[element]
                        tfev = timer_for_each_voilators[element]
                        if ttfev == tfev:
                            to_delete.append(element)
                if to_delete:
                    [timer_for_each_voilators.pop(key) for key in to_delete] 
            temp_timer_for_each_voilator.clear()
            temp_timer_for_each_voilator = timer_for_each_voilators.copy()
            start_time_for_del = 0

        if full_screen == True:
            cv2.namedWindow("Final_Output", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Final_Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Final_Output", frame)
        else:
            cv2.imshow("Final_Image", frame)

        key = cv2.waitKey(10) & 0xFF

        if video_path == "0" or video_path.startswith("rtsp") or video_path.startswith("http") or video_path.startswith("https") or video_path.startswith("tcp") or video_path.startswith("udp"):
            continue
        else:
        # if video_path != "0":
            if output_video_1 is None:
                fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                output_video_1 = cv2.VideoWriter(r"F:\Gopal\Coding\20201009\SingleSystem\COVID-19_Social_Distance_Maintainer\output_video\OutputVideo.avi", fourcc1, 19,(frame.shape[1], frame.shape[0]), True)
            elif output_video_1 is not None:
                output_video_1.write(frame)

            # Break the loop
            if key == ord("q"):
                break
        to_delete = f"./Frame.jpg"
        if path.exists(to_delete):
            os.remove(to_delete)
    return True
        
def check_on_main_zone(people_inside,frame):
    global social_distance_voilators, people_positions
    for p in people_inside:
        distances = []
        for n,sp in enumerate(sorted_points,1):
            x1 = p[0]
            y1 = p[1]
            x2 = sp[0]
            y2 = sp[1]
            dist = int(calculateDistance(x1, y1, x2, y2))
            distances.append([n,dist])
        sorted_distances = sorted(distances , key=lambda k: k[1])
        people_positions.append([sorted_distances[0][0], p])

    people_at_same_position = [list(item[1]) for item in groupby(sorted(people_positions), key=lambda x: x[0])]
    for psp in people_at_same_position:
        if len(psp)>1:
            for i in psp:
                social_distance_voilators.append(i[1])

    for pp in people_positions:
        if pp[1] not in social_distance_voilators:
            ground_point = pp[1]
            person_standing_at = pp[0]
            for ndc in nearest_distance_combo:
                if ndc[0] == person_standing_at:
                    for i in ndc[1]:
                        for link in i[0]:
                            if link != person_standing_at:
                                for j in people_positions:
                                    if link in j:
                                        #ground point of human detected
                                        m = ground_point
                                        #Another human near the detected one based on position.
                                        n = j[1]
                                        x1 = m[0]
                                        y1 = m[1]
                                        x2 = n[0]
                                        y2 = n[1]
                                        try:
                                            distance_between_pair = int(calculateDistance(x1, y1, x2, y2))
                                            distance_to_compare = int(i[1]*inside_main_zone)
                                            # print(f"distance_to_compare:{distance_to_compare}")
                                            if distance_between_pair<distance_to_compare:
                                                if m not in social_distance_voilators:
                                                    social_distance_voilators.append(m)
                                                if n not in social_distance_voilators:
                                                    social_distance_voilators.append(n)
                                        except:
                                            continue


def find_voilators(x,y,enum,multiplier):
    x1 = x[0]
    y1 = x[1]
    x2 = y[0]
    y2 = y[1]
    try:
        if enum>=len(zone_with_distances):
            enum = 0
        distance_between_pair = int(calculateDistance(x1,y1,x2,y2))
        distance_to_compare = int(zone_with_distances[enum][1]*multiplier)
        if distance_between_pair<distance_to_compare:
            if x not in social_distance_voilators:
                social_distance_voilators.append(x)
            if y not in social_distance_voilators:
                social_distance_voilators.append(y)
    except:
        pass


def red_box(red_boxes, frame):
    for i,items in enumerate(red_boxes):
        first_point = red_boxes[i][0]
        second_point = red_boxes[i][1]
        third_point = red_boxes[i][2]
        fourth_point = red_boxes[i][3]
        cv2.rectangle(frame,(second_point,first_point),(fourth_point,third_point),COLOR_RED,2)

def get_human_box_detection(boxes,scores,classes,height,width):
	""" 
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes : all our boxes coordinates
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
	# print(boxes)
	array_boxes = list() # Create an empty list
	for i in range(boxes.shape[1]):
		# If the class of the detected object is 1 and the confidence of the prediction is > 0.75
		if int(classes[i]) == 0 and scores[i] > 0.2:
			# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
			# To transform the box value into pixel coordonate values.
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes

def play_warning():
    try:  
        # print("Playing warning here.") 
        sd.play(data, fs)
        status = sd.wait()
        # print("Warning Played...")
        # print()
        good_to_write = True
    except:
        good_to_write = False
        print("Exception occured. Something wrong 3.")

    

def waiting_time():
    global time_to_wait
    for i in range(audio_length+waits+1):
        time.sleep(1)
        time_to_wait += 1
    time_to_wait=0


def call_perform_sdc(vp, s, s_file, afl):
    #globalizing variables
    global video_path, seconds, audio_path, audio_length, data, fs

    video_path = vp
    seconds = s
    audio_path = s_file
    audio_length = int(afl)
    # print(f"Audio length: {audio_length}")

    #Getting audio
    data, fs = sf.read(audio_path, dtype='float32')

    try:
        app.run(main)
    except SystemExit:
        pass