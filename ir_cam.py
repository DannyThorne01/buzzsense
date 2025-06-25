
from libcamera import Transform
from picamera2 import Picamera2
from time import sleep
import cv2, time
import numpy as np
from fastiecm import fastiecm

picam2 = Picamera2()
config = picam2.create_preview_configuration(
			main={"format":"RGB888", "size":(640,480)},
			transform=Transform(rotation=0))
picam2.configure(config)
picam2.start()
time.sleep(2)

state = picam2.capture_metadata()
picam2.set_controls({
	"AeEnable" : 0,
	"AwbEnable" : 0,
	"AnalogueGain": state["AnalogueGain"],
	"ExposureTime":state["ExposureTime"],
	"ColourGains":state["ColourGains"]
})
def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (r.astype(float) - b) / bottom
    return ndvi
def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min
    return out   
    
while True:
	frame = picam2.capture_array("main")     
	ndvi   = calc_ndvi(frame)
	contrast = contrast_stretch(ndvi)
	color = cv2.applyColorMap(contrast.astype(np.uint8), fastiecm)
	cv2.imshow("IR FRAMES - NDVI", color)
	if cv2.waitKey(1) & 0xFF == 27:
		break

