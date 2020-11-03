import pytest
import sys
sys.path.append("..")
from pre_processing import *

@pytest.mark.parametrize("video_name, result",[
("", "./input_video/GridTest5.mp4"),
("WebCam", "0"),
("./input_video/GridTest5.mp4","./input_video/GridTest5.mp4")
])
def test_select_video(video_name,result):
    assert select_video(video_name)==result


def test_wait_to_play_warning():
    assert wait_to_play_warning("5")==5
    assert wait_to_play_warning("10")==10
    assert wait_to_play_warning("15")==15
    assert wait_to_play_warning("20")==20
    assert wait_to_play_warning("25")==25
    assert wait_to_play_warning("30")==30
    assert wait_to_play_warning("35")==35
    assert wait_to_play_warning("40")==40
    assert wait_to_play_warning("45")==45
    assert wait_to_play_warning("50")==50
    assert wait_to_play_warning("55")==55
    assert wait_to_play_warning("60")==60

@pytest.mark.parametrize("audio_name, result",[
("", "./sound/covid_msg.wav"),
("./sound/covid_msg2.wav","./sound/covid_msg2.wav")
])
def test_select_audio(audio_name, result):
    assert select_audio(audio_name)==result


