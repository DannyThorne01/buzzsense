import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


yolo = YOLO('/home/danielthorne/best_yolov8n_ncnn_model')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640 , 480, rs.format.z16, 30)
# Start streaming
pipeline.start(config)
frame_count =0 
latest_boxes = []
inference_interval = 10
colour = (0, 255, 0)
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame(1)
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame or not ir_frame:
            continue

        
        color_image = np.asanyarray(color_frame.get_data())
        color_img_mdl= cv2.resize(color_image, (640, 480))
        det = [[], []]  
        lbls = [[], []] 
        if color_image.any():
                frame_count += 1
                if frame_count % inference_interval == 0: 
                        print("Processing frames...")
                        output_img_mdl = yolo.predict(color_img_mdl)
                        
                        latest_boxes.clear()
        
                        for result in output_img_mdl:
                                if result.boxes.shape[0] > 0:
                                    det[0].append(result.boxes.xyxy.cpu().numpy())
                                    lbls[0].append(result.boxes.cls.cpu().numpy())
                                for box in result.boxes:
                                    if box.conf[0] > 0.25:
                                        latest_boxes.append({
                                            "box": box.xyxy[0].cpu().numpy(),
                                            "label": "flower",
                                            "conf": float(box.conf[0])
                                        })
                if(len(latest_boxes) > 0 ):
                    print("Camera detected a flower !")
                    
                for det in latest_boxes:
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(color_img_mdl, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(color_img_mdl, f'{det["label"]} {det["conf"]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)    
                cv2.imshow('IR Stream (Left)', color_img_mdl)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
