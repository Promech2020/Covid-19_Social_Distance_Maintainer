import cv2
import re
import math
import imutils
import numpy as np
from colors import bcolors
from itertools import combinations
from help_functions import *
from get_points import get_markings

show_points_and_lines = True

markings = []
corner_points = []
input_data = dict()
n_vertical = 3
n_horizontal = 3

def get_essential_data():
    global markings, corner_points, input_data, n_vertical, n_horizontal
    get_markings()

    ######################################### 
    # Load the config for the top-down view #
    #########################################
    print(bcolors.WARNING +"[ Loading config file for drawing grid] "+ bcolors.ENDC)

    input_data = eval(open("./SupportingFiles/corner_points.txt", 'r').read())

    input_markings = list(input_data.values())
    for count,id in enumerate(input_markings[:-2],1):
        markings.append(input_data[f"P{count}"])

    #Get number of people horizontal and vertical view
    n_vertical = input_data["r"]
    n_horizontal = input_data["c"]


    marks_in_top = markings[:n_horizontal]
    marks_in_bottom = markings[-n_horizontal:]

    corner_points.append(marks_in_top[0])
    corner_points.append(marks_in_top[-1])
    corner_points.append(marks_in_bottom[-1])
    corner_points.append(marks_in_bottom[0])

    print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )

    img_path = "./SupportingImages/for_grid.bmp"
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=1280)

    a = tuple(corner_points[0])
    b = tuple(corner_points[1])
    d = tuple(corner_points[2])
    c = tuple(corner_points[3])

    #Corner points are organized as Red, Green, Blue and Black dots
    if show_points_and_lines == True:
        cv2.line(img, a, b, (0,0,0),2)
        cv2.line(img, a, c, (0,0,0),2)
        cv2.line(img, d, b, (0,0,0),2)
        cv2.line(img, d, c, (0,0,0),2)

        cv2.circle(img, a,5, (0,0,255), -1)
        cv2.circle(img, b,5, (0,100,0), -1)
        cv2.circle(img, c,5, (255,0,0), -1)
        cv2.circle(img, d,5, (120,7,80), -1)
        cv2.circle(img, a,10, (0,0,255), 2)
        cv2.circle(img, b,10, (0,100,0), 2)
        cv2.circle(img, c,10, (255,0,0), 2)
        cv2.circle(img, d,10, (120,7,80), 2)

    #Find outer corner points
    outer_corner1 = []
    outer_corner2 = []
    outer_corner3 = []
    outer_corner4 = []

    #Three points in outer corner1
    outer_corner1.append([0,corner_points[0][1]])
    outer_corner1.append([0,0])
    outer_corner1.append([corner_points[0][0],0])

    #Three points in outer corner2
    outer_corner2.append([corner_points[1][0],0])
    outer_corner2.append([1280,0])
    outer_corner2.append([1280,corner_points[1][1]])

    #Three points in outer corner3
    outer_corner3.append([1280,corner_points[2][1]])
    outer_corner3.append([1280,720])
    outer_corner3.append([corner_points[2][0],720])
    
    #Three points in outer corner4
    outer_corner4.append([corner_points[3][0],720])
    outer_corner4.append([0,720])
    outer_corner4.append([0,corner_points[3][1]])

    outer_corner_points = outer_corner1 + outer_corner2 + outer_corner3 + outer_corner4 
    # print(outer_corner_points)
    
    #Outer corner zones
    outer_corner_zones = []
    for n,each in enumerate(corner_points):
        if n == 0:
            ocz1 = [outer_corner_points[n],each, outer_corner_points[n+1]]
            ocz2 = [outer_corner_points[n+1],each, outer_corner_points[n+2]]
            outer_corner_zones.append(tuple(ocz1))
            outer_corner_zones.append(tuple(ocz2))
        elif n==1:
            n = n+2
            ocz1 = [outer_corner_points[n],each, outer_corner_points[n+1]]
            ocz2 = [outer_corner_points[n+1],each, outer_corner_points[n+2]]
            outer_corner_zones.append(tuple(ocz1))
            outer_corner_zones.append(tuple(ocz2))
        elif n == 2:
            n = n+4
            ocz1 = [outer_corner_points[n],corner_points[2], outer_corner_points[n+1]]
            ocz2 = [outer_corner_points[n+1],corner_points[2], outer_corner_points[n+2]]
            outer_corner_zones.append(tuple(ocz1))
            outer_corner_zones.append(tuple(ocz2))
        elif n == 3:
            n = n+6
            ocz1 = [outer_corner_points[n],corner_points[3], outer_corner_points[n+1]]
            ocz2 = [outer_corner_points[n+1],corner_points[3], outer_corner_points[n+2]]
            outer_corner_zones.append(tuple(ocz1))
            outer_corner_zones.append(tuple(ocz2))

    #Find number of zones available
    corner_zones = 8
    zone0 = 1
    boundary_zones = (n_horizontal+n_vertical-2)*2
    total_zones = zone0 + corner_zones + boundary_zones

    # cv2.imshow("progress", img)
    # cv2.waitKey(0)

    #Split boundary lines into numbers of people to find the points.
    points_on_ab = marks_in_top[1:-1]
    points_on_cd = marks_in_bottom[1:-1]

    points_between_corner_1_and_3 = []
    for mark in markings[1:]:
        if mark==corner_points[3]:
            break
        else:
            points_between_corner_1_and_3.append(mark)

    x_sorted_points_between_corner_1_and_3 = sorted(points_between_corner_1_and_3 , key=lambda k: k[0])
    x_sorted_points_on_ac = x_sorted_points_between_corner_1_and_3[:n_vertical-2]
    if len(x_sorted_points_on_ac)>=2:
        points_on_ac = sorted(x_sorted_points_on_ac , key=lambda k: k[1])
    else:
        points_on_ac = x_sorted_points_on_ac

    points_between_corner_2_and_4 = [mark for mark in markings[n_horizontal:-1]]
    x_sorted_points_between_corner_2_and_4 = sorted(points_between_corner_2_and_4 , key=lambda k: k[0], reverse=True)
    x_sorted_points_on_bd = x_sorted_points_between_corner_2_and_4[:n_vertical-2]
    if len(x_sorted_points_on_bd)>=2:
        points_on_bd = sorted(x_sorted_points_on_bd , key=lambda k: k[1])
    else:
        points_on_bd = x_sorted_points_on_bd

    points_on_outer_ab = []
    points_on_outer_cd = []
    points_on_outer_ac = []
    points_on_outer_bd = []
    #Find outer boundary points
    for i in points_on_ab:
        i_on_outer = []
        i_on_outer.append(i[0])
        i_on_outer.append(0)
        points_on_outer_ab.append(i_on_outer)
    for i in points_on_cd:
        i_on_outer = []
        i_on_outer.append(i[0])
        i_on_outer.append(720)
        points_on_outer_cd.append(i_on_outer)
    for i in points_on_ac:
        i_on_outer = []
        i_on_outer.append(0)
        i_on_outer.append(i[1])
        points_on_outer_ac.append(i_on_outer)
    for i in points_on_bd:
        i_on_outer = []
        i_on_outer.append(1280)
        i_on_outer.append(i[1])
        points_on_outer_bd.append(i_on_outer)

    sorted_points_on_outer_ab = sorted(points_on_outer_ab , key=lambda k: k[0])
    outer_ab = sorted_points_on_outer_ab
    outer_ab.insert(0,outer_corner1[2])
    outer_ab.append(outer_corner2[0])

    sorted_points_on_outer_bd = sorted(points_on_outer_bd , key=lambda k: k[1])
    outer_bd = sorted_points_on_outer_bd
    outer_bd.insert(0,outer_corner2[2])
    outer_bd.append(outer_corner3[0])

    sorted_points_on_outer_cd = sorted(points_on_outer_cd , key=lambda k: k[0], reverse= True)
    outer_cd = sorted_points_on_outer_cd
    outer_cd.insert(0,outer_corner3[2])
    outer_cd.append(outer_corner4[0])

    sorted_points_on_outer_ac = sorted(points_on_outer_ac , key=lambda k: k[1], reverse= True)
    outer_ac = sorted_points_on_outer_ac
    outer_ac.insert(0,outer_corner4[2])
    outer_ac.append(outer_corner1[0])

    outer_boundary = outer_ab + outer_bd + outer_cd + outer_ac

    sorted_points_on_ab = sorted(points_on_ab , key=lambda k: k[0])
    inner_ab = sorted_points_on_ab
    inner_ab.insert(0,corner_points[0])
    inner_ab.append(corner_points[1])

    sorted_points_on_bd = sorted(points_on_bd , key=lambda k: k[1])
    inner_bd = sorted_points_on_bd
    inner_bd.insert(0,corner_points[1])
    inner_bd.append(corner_points[2])

    sorted_points_on_cd = sorted(points_on_cd , key=lambda k: k[0], reverse=True)
    inner_cd = sorted_points_on_cd
    inner_cd.insert(0,corner_points[2])
    inner_cd.append(corner_points[3])

    sorted_points_on_ac = sorted(points_on_ac , key=lambda k: k[1], reverse=True)
    inner_ac = sorted_points_on_ac
    inner_ac.insert(0,corner_points[3])
    inner_ac.append(corner_points[0])

    
    inner_boundary = inner_ab + inner_bd + inner_cd + inner_ac

    outer_inner_connecter = list(zip(outer_boundary,inner_boundary))

    outer_inner_zones = []
    
    for n,each in enumerate(outer_inner_connecter):
        count = int(boundary_zones + (corner_zones/2) - 1)
        if n<count:
            zone = outer_inner_connecter[n] + outer_inner_connecter[n+1]
            outer_inner_zones.append(zone)
        else:
            zone = outer_inner_connecter[n] + outer_inner_connecter[0]
            outer_inner_zones.append(zone)

    for each in outer_inner_zones:
        for i in each:
            if each.count(i) == 2:
                outer_inner_zones.remove(each)
                break
            
    main_zone = [corner_points]
    cz1 = tuple([outer_corner_zones[1]])
    cz2 = tuple([outer_corner_zones[2]])
    cz3 = tuple([outer_corner_zones[3]])
    cz4 = tuple([outer_corner_zones[4]])
    cz5 = tuple([outer_corner_zones[5]])
    cz6 = tuple([outer_corner_zones[6]])
    cz7 = tuple([outer_corner_zones[7]])
    cz8 = tuple([outer_corner_zones[0]])
    top = tuple(outer_inner_zones[0:n_horizontal-1])
    right  = tuple(outer_inner_zones[n_horizontal-1:n_horizontal+n_vertical-2])
    bottom = tuple(outer_inner_zones[n_horizontal+n_vertical-2 : (n_horizontal*2)+n_vertical-3])
    left = tuple(outer_inner_zones[(n_horizontal*2)+n_vertical-3: (n_horizontal*2)+(n_vertical*2)-4])
    
    all_zones = tuple(main_zone) + cz1 + top  + cz2 + cz3 + right + cz4 + cz5  + bottom + cz6 + cz7 + left + cz8
    all_zones_tagged = dict()
    for n,az in enumerate(all_zones):
        if len(az)==4:
            # sorted_az = organize_points(list(az))
            # xy_sorted_az = sorted_az[:2] + [sorted_az[3]] + [sorted_az[2]]
            xy_sorted_az = organize_points(list(az))
        else:
            xy_sorted_az = organize_points(list(az))
        all_zones_tagged[f"Zone{n}"] = xy_sorted_az
        if show_points_and_lines == True:
            if len(xy_sorted_az)==4:
                resorted_az = (xy_sorted_az[0], xy_sorted_az[1], xy_sorted_az[3], xy_sorted_az[2])
                pts = np.array(resorted_az, np.int32) 
            else:
                pts = np.array(xy_sorted_az, np.int32) 
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (120,25,50),2) 

        if len(xy_sorted_az) == 4:
            i = xy_sorted_az[0]
            j = xy_sorted_az[-1]
            x1 = i[0]
            y1 = i[1]
            x2 = j[0]
            y2 = j[1]
            center = int((x1+x2)/2), int((y1+y2)/2)
            if show_points_and_lines == True:
                # cv2.circle(img, center,5, (0,0,255), -1)
                cv2.putText(img, f"Z{n}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        else:
            i = xy_sorted_az[0]
            j = xy_sorted_az[-1]
            x1 = i[0]
            y1 = i[1]
            x2 = j[0]
            y2 = j[1]
            first_center = int((x1+x2)/2), int((y1+y2)/2)
            k = xy_sorted_az[1]
            l = first_center
            a1 = k[0]
            b1 = k[1]
            a2 = l[0]
            b2 = l[1]
            center = int((a1+a2)/2), int((b1+b2)/2)
            if show_points_and_lines == True:
                # cv2.circle(img, center,3, (0,0,255), -1)
                cv2.putText(img, f"Z{n}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


    #Getting zones and their neighour zones(left and right)
    neighbour_zones_combo = []
    for n,zone in enumerate(all_zones[1:],1):
        if n==1:
            temp_list = []
            neighbours = [all_zones[-1],all_zones[2]]
            temp_list.append(zone)
            temp_list.append(neighbours)
            neighbour_zones_combo.append(temp_list)
        elif n == len(all_zones[1:]):
            temp_list = []
            neighbours = [all_zones[n-1],all_zones[1]]
            temp_list.append(zone)
            temp_list.append(neighbours)
            neighbour_zones_combo.append(temp_list)
        else:
            temp_list = []
            neighbours = [all_zones[n-1],all_zones[n+1]]
            temp_list.append(zone)
            temp_list.append(neighbours)
            neighbour_zones_combo.append(temp_list)

    #Combining all points in boundary except the corner points. Drawing white circles on those points.
    boundary_points = points_on_ab + points_on_cd + points_on_ac + points_on_bd
    for each in boundary_points:
        if show_points_and_lines == True:
            cv2.circle(img, tuple(each),3, (255,255,255), -1)
            cv2.circle(img, tuple(each),8, (255,255,255), 2)

    #Finding out coordinate values inside the boundary box.
    points_inside_boundary = [mark for mark in markings if mark not in boundary_points and mark not in corner_points]
    # print(points_inside_boundary)

    #Positions at boundary:
    boundary_positions = []

    #Combining all points acquired and sorting them.
    all_points = corner_points + points_on_ab + points_on_cd + points_on_ac + points_on_bd + points_inside_boundary
    y_sorted_points = sorted(all_points , key=lambda k: [k[1], k[0]])
    all_points_sorted = []
    new_list = [y_sorted_points[i:i+n_horizontal] for i in range(0, len(y_sorted_points), n_horizontal)]
    for each in new_list:
        sublist_sorted = sorted(each , key=lambda k: k[0])
        for data in sublist_sorted:
            all_points_sorted.append(data)
    # print(all_points_sorted)
    #Drawing counts in points to see the result of sorting.
    for i, data in enumerate(all_points_sorted):
        if data in corner_points or data in boundary_points:
            temp_list = []
            temp_list.append(data)
            temp_list.append(i+1)
            boundary_positions.append(temp_list)
        if show_points_and_lines == True:
            cv2.putText(img, str(i+1), tuple(data), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    
    positions_sorted = []
    all_boundary_points = boundary_points + corner_points
    # print(all_boundary_points)
    for n,az in enumerate(all_zones[1:]):
        temp_list = []
        for coordinate in az:
            if coordinate in all_boundary_points:
                for bp in boundary_positions:
                    if bp[0]==coordinate:
                        temp_list.append(bp[1])
        positions_sorted.append(temp_list)

    neighbour_zones_position_combo = list(zip(neighbour_zones_combo,positions_sorted))

    #Getting number of positions people can stay
    positions = [i+1 for i in range(n_horizontal*n_vertical)]
    # print(positions)

    #Getting combination of pairs between positions
    comb = list(combinations(positions, 2))
    distance_combo = []
    for i in comb:
        p = all_points_sorted[i[0]-1]
        q = all_points_sorted[i[1]-1]
        x1 = p[0]
        y1 = p[1]
        x2 = q[0]
        y2 = q[1]
        d = calculateDistance(x1, y1, x2, y2)
        distance_combo.append([i,int(d)])

    # print(distance_combo[:40])
    #Getting boundary lines distances:
    inner_boundary_points = []
    for each in inner_boundary:
        if each not in inner_boundary_points:
            inner_boundary_points.append(each)
    inner_boundary_distances = []
    for n,points in enumerate(inner_boundary_points):
        if n != len(inner_boundary_points)-1:
            i = points
            j = inner_boundary_points[n+1]
            x1 = i[0]
            y1 = i[1]
            x2 = j[0]
            y2 = j[1]
            dist = int(calculateDistance(x1,y1,x2,y2))
            inner_boundary_distances.append(dist)
        else:
            i = points
            j = inner_boundary_points[0]
            x1 = i[0]
            y1 = i[1]
            x2 = j[0]
            y2 = j[1]
            dist = int(calculateDistance(x1,y1,x2,y2))
            inner_boundary_distances.append(dist)
    
    #Dividing zones into top right bottom left
    top_zones = cz1 + top + cz2
    right_zones = cz3 + right + cz4
    bottom_zones = cz5 + bottom + cz6
    left_zones = cz7 + left + cz8

    #Dividing boundary distances into top right bottom left
    top_distances = tuple(inner_boundary_distances[0:n_horizontal-1])
    right_distances = tuple(inner_boundary_distances[n_horizontal-1:n_horizontal+n_vertical-2])
    bottom_distances = tuple(inner_boundary_distances[n_horizontal+n_vertical-2 : (n_horizontal*2)+n_vertical-3])
    left_distances = tuple(inner_boundary_distances[(n_horizontal*2)+n_vertical-3: (n_horizontal*2)+(n_vertical*2)-4])

    zones_with_distances = []
    #For top zones and distances
    for n,zones in enumerate(top_zones,1):
        if n==1:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(top_distances[0])
            zones_with_distances.append(temp_list)
        elif n==len(top_zones):
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(top_distances[-1])
            zones_with_distances.append(temp_list)
        else:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(top_distances[n-2])
            zones_with_distances.append(temp_list)
    #For right zones and distances
    for n,zones in enumerate(right_zones,1):
        if n==1:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(right_distances[0])
            zones_with_distances.append(temp_list)
        elif n==len(right_zones):
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(right_distances[-1])
            zones_with_distances.append(temp_list)
        else:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(right_distances[n-2])
            zones_with_distances.append(temp_list)
    #For bottom zones and distances
    for n,zones in enumerate(bottom_zones,1):
        if n==1:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(bottom_distances[0])
            zones_with_distances.append(temp_list)
        elif n==len(bottom_zones):
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(bottom_distances[-1])
            zones_with_distances.append(temp_list)
        else:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(bottom_distances[n-2])
            zones_with_distances.append(temp_list)
    #For left zones and distances
    for n,zones in enumerate(left_zones,1):
        if n==1:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(left_distances[0])
            zones_with_distances.append(temp_list)
        elif n==len(left_zones):
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(left_distances[-1])
            zones_with_distances.append(temp_list)
        else:
            temp_list = []
            for k,v in all_zones_tagged.items():
                if sorted(v) == sorted(list(zones)):
                    z = re.findall(r'\d+', k)
                    temp_list.append(int(z[0]))
                    break
            temp_list.append(left_distances[n-2])
            zones_with_distances.append(temp_list)

    # nearest_distance_combo = []
    # for pos in positions:
    #     pos_distances = []
    #     for combo in distance_combo:
    #         if pos in combo[0]:
    #             pos_distances.append(combo)

    #     pos_dist = sorted(pos_distances , key=lambda k: k[1])

    #     if all_points_sorted[pos-1] in corner_points:
    #         near2 = pos_dist[:2]
    #         nearest_distance_combo.append([pos,near2])
    #     elif all_points_sorted[pos-1] in boundary_points:
    #         near3 = pos_dist[:3]
    #         nearest_distance_combo.append([pos,near3])
    #     else:
    #         near4 = pos_dist[:4]
    #         nearest_distance_combo.append([pos,near4])

    near_dist_combo = []
    for pos in positions:
        pos_distances = []

        for combo in distance_combo:
            if pos in combo[0]:
                pos_distances.append(combo)

        if all_points_sorted[pos-1] in corner_points:
            near_positions = []
            near_pos_with_dist = []
            print(f"Enter 2 near positions for mark {pos}:")
            for i in range(2):
                near_pos = int(input(f"Near{i+1}: "))
                near_positions.append(near_pos)
            for pos_d in pos_distances:
                for near_p in near_positions:
                    if near_p in pos_d[0]:
                        near_pos_with_dist.append(pos_d)
                        break 
            near_dist_combo.append([pos,near_pos_with_dist])
            print()        

        elif all_points_sorted[pos-1] in boundary_points:
            near_positions = []
            near_pos_with_dist = []
            print(f"Enter 3 near positions for mark {pos}")
            for i in range(3):
                near_pos = int(input(f"Near{i+1}: "))
                near_positions.append(near_pos)
            for pos_d in pos_distances:
                for near_p in near_positions:
                    if near_p in pos_d[0]:
                        near_pos_with_dist.append(pos_d)
                        break
            near_dist_combo.append([pos,near_pos_with_dist])   
            print()      

        else:
            print(f"Enter 4 near positions for mark {pos}")
            near_positions = []
            near_pos_with_dist = []
            for i in range(4):
                near_pos = int(input(f"Near{i+1}: "))
                near_positions.append(near_pos)
            for pos_d in pos_distances:
                for near_p in near_positions:
                    if near_p in pos_d[0]:
                        near_pos_with_dist.append(pos_d)
                        break
            near_dist_combo.append([pos,near_pos_with_dist]) 
            print()        

    cv2.imwrite("./Zones.jpg",img)
    background_calculation = dict()
    background_calculation["nearest_distance_combo"] = near_dist_combo
    background_calculation["all_points_sorted"] = all_points_sorted
    background_calculation["all_zones_tagged"] = all_zones_tagged
    background_calculation["zones_with_distances"] = zones_with_distances
    background_calculation["neighbour_zones_position_combo"] = neighbour_zones_position_combo
    with open('./SupportingFiles/background_data.txt', 'w') as outfile:
        print(background_calculation, file=outfile)
    


if __name__ == "__main__":
    get_essential_data()
