import argparse
from collections import defaultdict, deque

import cv2 as cv
from inference.models.utils import get_roboflow_model
import supervision as sv
import numpy as np

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])#[A point, B point, C point, D point]

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
        [0, 0], #A point
        [TARGET_WIDTH - 1, 0], #B point
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], #C point
        [0, TARGET_HEIGHT - 1], #D point
    ]
)

def parse_arguments()-> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Vehicle Speed Estimation using Inference and Supervision")
    parser.add_argument('--source_video_path', type=str, default='data/vehicles.mp4', help="Path to the source video file")
    return parser.parse_args()

class ViewTransformer:
    def __init__(self, source:np.ndarray, target:np.ndarray):
        source=source.astype(np.float32)
        target=target.astype(np.float32)
        self.m=cv.getPerspectiveTransform(source, target)
    def transform_points(self, points:np.ndarray) -> np.ndarray:
        reshaped_points=points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points=cv.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

if __name__ == "__main__":
    args=parse_arguments()
    video_info=sv.VideoInfo.from_video_path(args.source_video_path)
    model=get_roboflow_model("yolov8x-640")
    byte_track=sv.ByteTrack(frame_rate=video_info.fps)

    thickness=sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale=sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator=sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    polygon_zone=sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer=ViewTransformer(source=SOURCE, target=TARGET)

    coordinates=defaultdict(lambda: deque(maxlen=video_info.fps))#Dictionary that storage the past positions of the cars

    trace_annotator = sv.TraceAnnotator(thickness=thickness,trace_length=video_info.fps * 2,position=sv.Position.BOTTOM_CENTER,)

    for frame in frame_generator:
        result=model.infer(frame)[0]
        detections=sv.Detections.from_inference(result)
        detections=detections[polygon_zone.trigger(detections)]#Filter the detections to keep just the detections inside the polygon zone
        detections=byte_track.update_with_detections(detections=detections)
        points=detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)#Returns the bottom center point coordenates from the object bouding box
        points=view_transformer.transform_points(points=points).astype(int)#Transform the perspective of the points using the homographic matrix and turn it to integer

        labels=[]
        for tracker_id, [x, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)#Add the 'y' value in the key 'tracker_id' of the dictionary
            if len(coordinates[tracker_id]) < video_info.fps/2:#If there isn't sufficient data to calculate the speed...
                labels.append(f"#{tracker_id}")
            else:
                coordinates_start=coordinates[tracker_id][-1]#Get the oldest coordenate registered(start point)
                coordinates_end=coordinates[tracker_id][0]#Get the newest coordenate(end point)
                distance=abs(coordinates_start-coordinates_end)#Calculate the difference between them(distance)
                time=len(coordinates[tracker_id])/video_info.fps
                speed=(distance/time)*3.6
                #labels.append(f"#X: {x}, Y: {y}")
                labels.append(f"#{tracker_id} {int(speed)} km/h")
        annotated_frame = frame.copy()
        annotated_frame=sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        resized_frame = cv.resize(annotated_frame, (640, 440))
        cv.imshow("Video", resized_frame)
        if cv.waitKey(1)==ord("q"):
            break
