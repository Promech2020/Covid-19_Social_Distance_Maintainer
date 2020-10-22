import cv2
import numpy as np
import imutils
from help_functions import organize_points

 
# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])

video_name = input("Enter video path including video extension: ")
vs = cv2.VideoCapture(video_name)
# Loop until the end of the video stream
while True:    
    # Load the frame and test if it has reache the end of the video
    (frame_exists, frame) = vs.read()
    frame = imutils.resize(frame, width=1280)
    cv2.imwrite("./SupportingImages/for_grid.jpg",frame)
    break

# Create a black image and a window
windowName = 'MouseCallback'
cv2.namedWindow(windowName)

# Load the image 
img_path = "./SupportingImages/for_grid.jpg"
img = cv2.imread(img_path)

# Create an empty list of points for the coordinates
list_points = list()

# bind the callback function to window
cv2.setMouseCallback(windowName, CallBackFunc)


def get_markings():
    row_num = int(input("Enter number of rows: "))
    col_num = int(input("Enter number of people in each row: "))
    marks_count = row_num*col_num

    print()
    print("Click at the marks left to right followed by top to bottom.")
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(list_points) == marks_count:
            config_data = dict()
            for c,points in enumerate(list_points,1):
                config_data[f"P{c}"] = points

            config_data["r"] = row_num
            config_data["c"] = col_num
            # Write the result to the config file
            # with open('./SupportingFiles/corner_points_test.txt', 'w') as outfile:
            with open('./SupportingFiles/corner_points.txt', 'w') as outfile:
                print(config_data, file=outfile)
            break
        if cv2.waitKey(20) == 27:
            break
    print(points)
    cv2.destroyAllWindows()
