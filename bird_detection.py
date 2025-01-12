import cv2 as cv
# import datetime
import os
from ultralytics import YOLO

# loops over the bird image files
def path_list(path):
    directory = {'path': path,'categories': 0,'images': 0}
    file_path = []
    for dirpath,dirnames,filenames in os.walk(directory['path']):
        for filename in filenames:
            # Get the full file path
            path = os.path.join(dirpath, filename)
            file_path.append(path)
        directory['categories'] += len(dirnames)
        directory['images'] += len(filenames)
    print(f'there is {directory['categories']} subfolder classes and {directory['images']} images in total')
    return file_path

# i = 0
# while i < directory['images']-1:
#     img = cv.imread(file_path[i],-1)
#     cv.resize(img,(frame_width,frame_height))
#     cv.imshow('display', img)
#     i += 1
#     if cv.waitKey(30) == ord('q'):
#         break

# Only save an image on frame 0
def img(path,text):
    # Loading pretrained YOLO model (will be downloaded on first run)
    model = YOLO("model/yolov8n.pt", "v8")

    img = cv.imread(path,-1)

    # Do prediction on image, with confidence greater than 80%
    detect_params = model.predict(source=[img], conf=0.5, save=False)

    DP = detect_params[0].numpy()
    font = cv.FONT_HERSHEY_SIMPLEX

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            # clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            c = box.cls
            # Name of object detected (e.g. 'bird')
            class_name = model.names[int(c)]

        # If the class name contains the word 'bird', do something with the frame
        if 'bird' in class_name.lower():

            # Draw green rectangle around the object
            cv.rectangle(
                img,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                (0, 255, 0),
                3,
            )
            # Add some text labelling to the rectangle
            cv.putText(
                img,
                str(text),
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (0, 0, 255),
                2,
                lineType=cv.LINE_AA
            )
    else:
        cv.putText(
                img,
                str(text),
                (100,100),
                font,
                1.5,
                (0, 0, 255),
                3,
                lineType=cv.LINE_AA
            )

    # Display the frame onscreen
    cv.imshow("Object Detection", img)
    cv.waitKey(5000)
    cv.destroyAllWindows()
