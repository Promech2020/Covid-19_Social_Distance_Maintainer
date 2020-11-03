import pytest
import sys
sys.path.append("..")
from help_functions import *

@pytest.mark.parametrize("start,end,segment,result",[
((0,0), (100,100),4,[[25,25],[50,50],[75,75]])
])
def test_split(start,end,segment,result):
    assert split(start,end,segment)==result

@pytest.mark.parametrize("points,result",[
([[5,25],[50,5],[72,89],[33, 18]], [[50, 5], [33, 18], [5, 25], [72, 89]])
])
def test_sort_y(points, result):
    assert sort_y(points)==result


@pytest.mark.parametrize("points,result",[
([[5,25],[50,5],[72,89],[33, 18]], [[5, 25], [33, 18], [50, 5], [72, 89]])
])
def test_sort_x(points, result):
    assert sort_x(points)==result

@pytest.mark.parametrize("points,result",[
([[5,25],[50,5],[72,89],[33, 18]], [[33, 18], [50, 5], [5, 25], [72, 89]])
])
def test_organize_points(points, result):
    assert organize_points(points)==result

@pytest.mark.parametrize("line1,line2,result",[
(((5,25),(50,5)),((72,89),(33, 18)),[30, 13]),
(((133, 108), (250, 65)), ((85, 200), (372, 189)),[-140, 208])
])
def test_line_intersection(line1, line2, result):
    assert line_intersection(line1, line2)==result

@pytest.mark.parametrize("x1,y1,x2,y2,result",[
    (50,25,98,166, 148),
    (78,225,198,102, 171),
    (92,215,701,368, 627),
    (73,125,910,196, 840),
    (150,625,308,216, 438),
])
def test_calculate_distance(x1,y1,x2,y2,result):
    assert calculateDistance(x1, y1, x2, y2)==result

@pytest.mark.parametrize("array,result",[
    ([(351, 576, 716, 717), (261, 546, 488, 699), (130, 527, 446, 660)], ([(646, 533), (622, 374), (593, 288)], [(646, 716), (622, 488), (593, 446)])),
    # ([(290, 129, 716, 340), (352, 571, 717, 712), (270, 555, 473, 698), (461, 1069, 720, 1279)],()),
    # ([(287, 105, 699, 318), (368, 576, 720, 714), (251, 550, 495, 698), (459, 1067, 720, 1279)],()),
    # ([(278, 129, 699, 325), (255, 554, 454, 698), (365, 556, 720, 699), (453, 1071, 720, 1278)],()),
    # ([(145, 497, 475, 610), (299, 556, 720, 727), (140, 616, 366, 714), (454, 1069, 720, 1280)],()),
    # ([(143, 687, 485, 819), (142, 303, 467, 420), (148, 609, 477, 710), (458, 1070, 720, 1278)],())
])
def test_get_centroids_and_groundpoints(array, result):
    assert get_centroids_and_groundpoints(array)==result


@pytest.mark.parametrize("box,result",[
    ((456, 1069, 720, 1278),((1173, 588), (1173, 720))),
    ((147, 460, 479, 611),((535, 313), (535, 479))),
    ((295, 544, 720, 729),((636, 507), (636, 720))),
    ((305, 134, 702, 330),((232, 503), (232, 702))),
    ((268, 557, 468, 699),((628, 368), (628, 468)))
])
def test_get_points_from_box(box,result):
    assert get_points_from_box(box)==result