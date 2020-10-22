import time
import copy
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
from helping_functions import image_resize, create_blank, get_human_box_detection, get_centroids, get_points_from_box, rescale_image
from final_windows import Final
from vidgear.gears import CamGear
import sounddevice as sd
import soundfile as sf
import itertools
import threading
from threading import Lock
import imutils
import math
import yaml
import sys
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 320, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

#Defining red color rgb value
COLOR_RED = (0, 0, 255)

#Dictionary to save distance between pairs
distance_between_pairs = dict()
temp_dict_for_distance_between_pairs = dict()

#Dictionary to start timer when the distance between pairs is less than the minimum distance defined.
timer_for_each_pairs = dict()
temp_dict_for_timer_for_each_pairs = dict()

#Current time condition to wait between warnings.
time_to_wait = 0

#To save latest frames from webcam/cctv
latest_frame = None
last_ret = None
loc = Lock()

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)


    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

   #Getting audio
    data, fs = sf.read(FLAGS.sound_file, dtype='float32')

    input_size = FLAGS.size
    video_path = FLAGS.video
    # print(video_path)
    # print()
    # print(f"Minimum Distance: {FLAGS.minimum_distance}")
    # print()
    global temp_dict_for_distance_between_pairs, temp_dict_for_timer_for_each_pairs, time_to_wait
    global latest_frame, loc, last_ret
    start = time.time()
    flags.DEFINE_float('starting_time', start, 'time when execution starts')
    
    good_to_run = False
    good_to_write = True
    output_video_1 = None

    ######################################################
    # 				START THE VIDEO STREAM               #
    ######################################################

    def rtsp_cam_buffer(vs):
        global latest_frame, loc, last_ret
        while True:
            with loc:
                if vs.isOpened():
                    last_ret, latest_frame = vs.read()

    if video_path == "0" or video_path.startswith("rtsp"):
        try:
            vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
            if frame_per_seconds == 0:
                print("WebCam input is not right. Please try again.")
                os.exit(0)
            t1 = threading.Thread(target=rtsp_cam_buffer,args=(vs,),name="webcam thread")
            t1.daemon=True
            t1.start()
            good_to_run = True
            good_to_write = True
        except:
            good_to_run = False
            good_to_write = False
            end = time.time()
            time_elapsed = int(end - FLAGS.starting_time)

    elif video_path.startswith("http"):
        try:
            vs = CamGear(source=video_path, y_tube =True,  time_delay=1, logging=True).start() 
            frame_per_seconds = vs.framerate
            good_to_run = True
            good_to_write = True

        except:
            vs = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
            if frame_per_seconds == 0:
                print("WebCam input is not right. Please try again.")
                os._exit(0)
            t1 = threading.Thread(target=rtsp_cam_buffer,args=(vs,),name="webcam thread")
            t1.daemon=True
            t1.start()
            good_to_run = True
            good_to_write = True
            
    else:	
        try:
            vs = cv2.VideoCapture(video_path)
            frame_per_seconds = int(vs.get(cv2.CAP_PROP_FPS))
            good_to_run = True
            good_to_write = True
        except:
            good_to_run = False
            good_to_write = False
            end = time.time()
            time_elapsed = int(end - FLAGS.starting_time)
            print(f"Time consumed: {time_elapsed} seconds.")
            print("Webcam not connected.")
            os._exit(0)

    start_timer = 0
    l_count = 0
    skip_count = 0
    # Loop until the end of the video stream
    frame_to_check = int(frame_per_seconds/20)
    fps_count = 0

    box_and_centroid = []
    close_pairs = []
    flat_list = []
    common_close_pairs = []
    boxes_to_make_red = []
    total_time = 0
    while True and good_to_run == True:	
        if start_timer == 0:
            start_timer = time.time()
        if video_path == "0" or video_path.startswith("rtsp"):
            # try:
            if (last_ret is not None) and (latest_frame is not None):
                frame = latest_frame.copy()
                print(f"Loop {l_count} is Ok. Skipped {skip_count}")
                skip_count = 0
            else:
                print(f"Loop {l_count} is Skipped.")
                skip_count += 1
                l_count += 1
                if skip_count > 10:
                    # print("Network Problem. Please restart the system.")
                    # os._exit(0)
                    distance_between_pairs.clear()
                    timer_for_each_pairs.clear()
                    temp_dict_for_distance_between_pairs.clear()
                    temp_dict_for_timer_for_each_pairs.clear()
                    time_to_wait = 0
                    latest_frame = None
                    last_ret = None
                    print("Something went wrong.Restarting...")
                    # continue
                    
                time.sleep(0.2)
                continue 
            good_to_run = True
            good_to_write = True
        

            # except:
            #     good_to_run = False
            #     good_to_write = False
            #     end = time.time()
            #     time_elapsed = int(end - FLAGS.starting_time)
            #     print(f"Time consumed: {time_elapsed} seconds.")
            #     if video_path=="0":
            #         print("WebCam stopped working.")
            #     else:
            #         print("IP Camera stopped working.")
            #     os._exit(0)          
        elif video_path.startswith("http"):
            try:
                frame = vs.read()
                good_to_run = True
                good_to_write = True
            except:
                good_to_run = False
                good_to_write = False
                end = time.time()
                time_elapsed = int(end - FLAGS.starting_time)
                print(f"Time consumed: {time_elapsed} seconds.")
                print("Online video link stopped working.")
                os._exit(0)
        else:
            try:
                (frame_exist, frame) = vs.read()
                good_to_run = True
                good_to_write = True
            except:
                good_to_run = False
                good_to_write = False
                end = time.time()
                time_elapsed = int(end - FLAGS.starting_time)
                print(f"Time consumed: {time_elapsed} seconds.")
                print("Could not get frames from video.")
                os._exit(0)
            
        if frame is None:
            # end = time.time()
            # final_time = end - start_timer
            # print(f"Total Time Consumed: {final_time}")
            # print(f"Total Time Consumed: {total_time}")
            break
        else:
            if fps_count == 0 or fps_count%frame_to_check==0:
                # event.set()
                good_to_run = True
                good_to_write = True

                box_and_centroid.clear()
                close_pairs.clear()
                flat_list.clear()
                common_close_pairs.clear()
                boxes_to_make_red.clear()
                # Resize the image to the correct size
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = image_resize(frame, width = FLAGS.frame_size)

                # frame_size = frame.shape[:2]
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
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if len(boxes)>0:	
                    # Get the human detected in the frame and return the 2 points to build the bounding box  
                    array_boxes_detected = get_human_box_detection(final_boxes,final_scores[0].tolist(),final_classes[0].tolist(),frame.shape[0],frame.shape[1])

                    if len(array_boxes_detected)>0:
                        # Both of our lists that will contain the centroïds coordonates and the ground points
                        array_centroids = get_centroids(array_boxes_detected)
                        box_and_centroid = list(zip(array_centroids,array_boxes_detected))

                        # Check if 2 or more people have been detected (otherwise no need to detect)
                        if len(array_centroids) >= 2:
                            for i,pair in enumerate(itertools.combinations(array_centroids, r=2)):
                                # Check if the distance between each combination of points is less than the minimum distance chosen
                                distance_between_pair = math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 )
                                # print(distance_between_pair)	
                                #Pairs with probability that will not maintain social distancing.
                                if distance_between_pair <= int(FLAGS.minimum_distance)*2:
                                    #Creating new dictionary containing distances between pairs
                                    distance_between_pairs[f"pairs{i}"] = distance_between_pair
                                    #Checking and creating timer for pairs from distance_between_pairs
                                    if f"pairs{i}" not in timer_for_each_pairs.keys():
                                        timer_for_each_pairs[f"pairs{i}"] = 0
                                    
                                if distance_between_pair < int(FLAGS.minimum_distance):
                                    close_pairs.append(pair)
                            
                            for sublist in close_pairs:
                                for item in sublist:
                                    flat_list.append(item)
                            common_close_pairs = list(set(flat_list))
                            # print(common_close_pairs)	
                            for ccp in common_close_pairs:
                                for b_and_c in box_and_centroid:
                                    if ccp == b_and_c[0]:
                                        boxes_to_make_red.append(b_and_c[1]) 
                            # print(boxes_to_make_red)
                            if boxes_to_make_red:
                                red_box(boxes_to_make_red, frame)
                            
            else:
                frame = image_resize(frame, width = FLAGS.frame_size)
                if boxes_to_make_red:
                    red_box(boxes_to_make_red, frame)

        fps_count += 1
        end_timer = time.time()
        current_time_value = end_timer - start_timer  
        # print()
        # print("Distance between pairs")
        # print(distance_between_pairs)
        # print()
        # print("Time between pairs")
        # print(timer_for_each_pairs)
        # print()
        # print("Time to wait")
        # print(time_to_wait)


        # print(time_to_wait)
        if current_time_value > 0.5:
            
            if len(temp_dict_for_distance_between_pairs) != 0:
                to_delete = []
                for element in distance_between_pairs:
                    if element in temp_dict_for_distance_between_pairs:
                        if temp_dict_for_distance_between_pairs[element] == distance_between_pairs[element]:
                            to_delete.append(element)
                if to_delete:
                    # print("Before Deletion")
                    # print("Original Dictionary")
                    # print(f"{distance_between_pairs}")
                    # print(f"{timer_for_each_pairs}")
                    # print("Temporary Dictionary")
                    # print(f"{temp_dict_for_distance_between_pairs}")
                    # print(f"{temp_dict_for_timer_for_each_pairs}")

                    [distance_between_pairs.pop(key) for key in to_delete] 
                    [timer_for_each_pairs.pop(key) for key in to_delete]

                    # print("After Deletion")
                    # print(f"{distance_between_pairs}")
                    # print(f"{timer_for_each_pairs}")            
            # print(distance_between_pairs)
            for key,value in distance_between_pairs.items():
                if value < float(FLAGS.minimum_distance):
                    timer_for_each_pairs[key] += current_time_value
                else:
                    timer_for_each_pairs[key] = 0
                    
            if len(distance_between_pairs)>0:
                t = timer_for_each_pairs.values()
                t_max = max(t)
                # print(t_max)
                if t_max >= FLAGS.seconds and time_to_wait==0:
                    t_max = 0
                    temp_dict_for_distance_between_pairs.clear()
                    temp_dict_for_distance_between_pairs = copy.deepcopy(distance_between_pairs)
                    # print()
                    # print("Distance between pairs")
                    # print(distance_between_pairs)
                    # print()
                    # print("Time between pairs")
                    # print(timer_for_each_pairs)
                    # print()
                    # print()
                    # print("Time to wait")
                    # print(time_to_wait)
                    # print("Playing warning here.......")
                    threading.Thread(target = play_warning, args = [data, fs]).start()
                    threading.Thread(target= waiting_time, args=[frame_per_seconds]).start()
                    
            start_timer = 0
        l_count += 1
        # # To check time consumed by each frame calculation.
        # curr_time = time.time()
        # exec_time = curr_time - prev_time
        # info = "time: %.2f ms" %(1000*exec_time)
        # total_time += exec_time
        # print(info)

        # cv2.namedWindow("Final_Output", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Final_Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Final_Output", frame)

        #Save images to folder.
        # filename = f"./output_image/frame{l_count}.png"
        # cv2.imwrite(filename, frame)
        key = cv2.waitKey(1) & 0xFF
        if video_path == "0" or video_path.startswith("http") or video_path.startswith("rtsp"):
            continue
        else:
        # if video_path != "0":
            if output_video_1 is None:
                fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
                output_video_1 = cv2.VideoWriter("./output_video/video.avi", fourcc1, int(frame_per_seconds),(frame.shape[1], frame.shape[0]), True)
            elif output_video_1 is not None:
                output_video_1.write(frame)

            # Break the loop
            if key == ord("q"):
                break

def red_box(red_boxes, frame):
    for i,items in enumerate(red_boxes):
        first_point = red_boxes[i][0]
        second_point = red_boxes[i][1]
        third_point = red_boxes[i][2]
        fourth_point = red_boxes[i][3]
        cv2.rectangle(frame,(second_point,first_point),(fourth_point,third_point),COLOR_RED,2)
    # return frame
def play_warning(d,f):
    try:    
        sd.play(d, f)
        status = sd.wait()
        # sd.stop()
        good_to_run = True
        good_to_write = True
    except:
        # eve.clear()
        good_to_run = False
        good_to_write = False
        end = time.time()
        time_elapsed = int(end - FLAGS.starting_time)
        print(f"Time consumed: {time_elapsed} seconds.")
        print("Could not play the sound.")
        os._exit(0)
    

def waiting_time(frame_per_seconds):
    # print()
    # print(f"frame per seconds {frame_per_seconds}")
    # print()
    global time_to_wait
    # print(time_to_wait)
    for i in range(int(FLAGS.audio_file_length)+FLAGS.waits):
        for j in range(int(frame_per_seconds*2)):
            to_sleep = 1/(frame_per_seconds*2)
            time.sleep(to_sleep)
            time_to_wait += to_sleep
    time_to_wait=0


def call_perform_sdc():
    # while True:
    try:
        app.run(main)
    except SystemExit:
        pass