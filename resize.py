import cv2

input_path="o_2.mp4"
output_path="o_3.mp4"

vidcap = cv2.VideoCapture(input_path)
success, image = vidcap.read()

height, width, layers = image.shape
new_h = int(height/2)
new_w = int(width/2)

out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), 15, (new_w,new_h))

while success:
    resize = cv2.resize(image, (new_w, new_h))
    success, image = vidcap.read()
    cv2.imshow('Conference Social distancing analyser', resize)
    cv2.waitKey(1)
    out.write(resize)