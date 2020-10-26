# Room safe capacity checker and face mask detection wrt COVID19 

## Backgroud
Inspired by [Social-Distancing-Analyser-COVID-19](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19) and [Face-Mask-Detection](https://github.com/balajisrinivas/Face-Mask-Detection)

## Pipeline
```
Video --> Moving Person Detection --> Faces Detection --> Wearing Mask Detection
```
## Requirements

Mask Detection Model:

+ [mask_detector.model](https://github.com/balajisrinivas/Face-Mask-Detection/blob/master/mask_detector.model)

Face Detection Model:

+ [face_detector](https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/face_detector)

Moving Target Model: 

+ [YOLOv3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

+ [YOLOv3.weights](https://pjreddie.com/media/files/yolov3.weights)

Or higher speed and lower accuarcy

+ [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
+ [yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)


## Features

For a video with people in the rooms:

+ Detect how many people in the conference room, if the numbers beyound the limit of COVID19 safety limit, warning.
+ Detect people wearing a mask or not, currently works well on front faces, but not good on side faces.


## Future

+ Improve face detection and make detection accuracy of side faces.
+ Fix repeate detection of faces under overlaped ROI.

