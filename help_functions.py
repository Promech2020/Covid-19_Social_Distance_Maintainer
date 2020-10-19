import math

def split(start, end, segments):
    x_delta = (end[0] - start[0]) / float(segments)
    y_delta = (end[1] - start[1]) / float(segments)
    points = []
    for i in range(1, segments):
        points.append([int(start[0] + i * x_delta), int(start[1] + i * y_delta)])
    return points

def sort_y(pts):
    pts.sort(key = lambda x: x[1]) 
    return pts 

def sort_x(pts):
    pts.sort(key = lambda x: x[0]) 
    return pts 
    
def organize_points(p):
    y_sorted = sort_y(p)
    x1 = y_sorted[:2]
    x2 = y_sorted[2:]
    x1_sorted = sort_x(x1)
    x2_sorted = sort_x(x2)
    sorted_points = x1_sorted + x2_sorted
    return sorted_points

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    pts = [x,y]
    return pts

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 
    
def get_centroids_and_groundpoints(array_boxes_detected):
    """
    For every bounding box, compute the centroid and the point located on the bottom center of the box
    @ array_boxes_detected : list containing all our bounding boxes 
    """
    array_centroids, array_groundpoints = [],[] # Initialize empty centroid and groundpoints.
    for index,box in enumerate(array_boxes_detected):
        # Get the centroid
        centroid,groundpoints = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(groundpoints)
    return array_centroids, array_groundpoints


def get_points_from_box(box):
    """
    Get the center of the bounding.
    @ param = box : 2 points representing the bounding box
    @ return = centroid (x1,y1)
    """
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(((box[1]+box[3])/2))
    center_y = int(((box[0]+box[2])/2))
    center = center_x, center_y
    ground = center_x, box[2]

    return center, ground