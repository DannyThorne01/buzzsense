import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort


# yolo = YOLO('/home/danielthorne/yolo_model.pt')
yolo = YOLO('/home/danielthorne/best_yolov8n_ncnn_model')
session = ort.InferenceSession('/home/danielthorne/yolo_int8.onnx',providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640 , 480, rs.format.z16, 30)
# Start streaming
pipeline.start(config)
frame_count =0 
latest_boxes_l = []
latest_boxes_r = []
inference_interval = 10  # Every 10 frames
colour = (0, 255, 0)
try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame(1)
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame or not ir_frame:
            continue

        # Convert to NumPy array
        color_image = np.asanyarray(color_frame.get_data())
        color_img_mdl= cv2.resize(color_image, (640, 480))
        ir_image = np.asanyarray(ir_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.5), cv2.COLORMAP_JET)
        ir_image = cv2.applyColorMap(cv2.convertScaleAbs(ir_image,alpha=0.5), cv2.COLORMAP_JET)
        det = [[], []]  # det[0] = left camera, det[1] = right camera
        lbls = [[], []] 
        # Show the frame
#        cv2.imshow('RealSense Color Stream', color_image)
        if color_image.any():
                frame_count += 1

                if frame_count % inference_interval == 0: 
                        print("Processing frames...")
                        
#                         image = color_img_mdl.astype(np.float32) / 255.0
#                         image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
#                         image = np.expand_dims(image, axis=0)
                        output_img_mdl = yolo.predict(color_img_mdl)
#                         output_img_mdl = session.run(None, {input_name: image})
                        print(output_img_mdl)
                        latest_boxes_l.clear()
                        latest_boxes_r.clear()
                        
                        for result in output_img_mdl:
                                if result.boxes.shape[0] > 0:
                                    det[0].append(result.boxes.xyxy.cpu().numpy())
                                    lbls[0].append(result.boxes.cls.cpu().numpy())
                                for box in result.boxes:
                                    if box.conf[0] > 0.25:
                                        latest_boxes_l.append({
                                            "box": box.xyxy[0].cpu().numpy(),
                                            "label": "flower",
                                            "conf": float(box.conf[0])
                                        })
                if(len(latest_boxes_l) > 0 ):
                    print("Camera detected a flower !")
                    
                for det in latest_boxes_l:
                    x1, y1, x2, y2 = map(int, det["box"])
                    cv2.rectangle(color_img_mdl, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(color_img_mdl, f'{det["label"]} {det["conf"]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
#                         
                        
                cv2.imshow('IR Stream (Left)', color_img_mdl)
#        cv2.imshow('RealSense Depth Stream', depth_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
