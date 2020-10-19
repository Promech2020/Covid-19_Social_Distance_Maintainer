from perform_sdc import call_perform_sdc
import scipy.io.wavfile as wav
import time
import sys
import os
import cv2

def pre_process(v_path, wait_time_before, a_path):

	video_path = select_video(v_path)

	seconds = wait_to_play_warning(wait_time_before)

	soundfile = select_audio(a_path)

	# #Get length of audio file
	(source_rate, source_sig) = wav.read(soundfile)
	audio_file_length = len(source_sig) / float(source_rate)

	call_perform_sdc(video_path, seconds, soundfile, audio_file_length)

######################################### 
#		     Select the video 			#
#########################################
def select_video(video_name):
	if video_name == "":
		video_p=r"F:\Gopal\Coding\20201009\SingleSystem\COVID-19_Social_Distance_Maintainer\input_video\GridTest5.mp4" 
	elif video_name == "WebCam":
		video_p = "0"
	else :
		video_p = video_name
	return video_p

######################################### 
#		    Time to wait			#
#########################################
def wait_to_play_warning(sec):
	#Take input for how many seconds do you want to wait when two people are close enough
	seconds = int(sec)
	return seconds

######################################### 
#		    Select Audio File		#
#########################################
def select_audio(audio):
	#Take input for how many seconds do you want to wait after playing warning.
	if audio == "":
		sound = r"F:\Gopal\Coding\20201009\SingleSystem\COVID-19_Social_Distance_Maintainer\sound\covid_msg.wav"
		# sound = "../sound/covid_msg.wav"
	else:
		sound = audio
	return sound
	



