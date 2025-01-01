import os
import json
import cv2
from collections import OrderedDict

class Labelme2YOLOSegmentation(object):
    
    def __init__(self, json_dir):
        self._json_dir = json_dir
        self._label_id_map = self._get_label_id_map(self._json_dir)

    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        # Iterate over each JSON file in the directory to extract all unique labels
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        # Map each label to a unique id
        return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])

    def convert(self):
        json_names = [file_name for file_name in os.listdir(self._json_dir) 
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and file_name.endswith('.json')]

        for json_name in json_names:
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))
            
            print(f'Converting {json_name} ...')

            # Get YOLO segmentation objects
            yolo_obj_list = self._get_yolo_object_list(json_data)

            # Save YOLO segmentation label file
            self._save_yolo_label(json_name, yolo_obj_list)

    def _get_yolo_object_list(self, json_data):
        yolo_obj_list = []
        
        # Get image dimensions (height, width)
        img_path = os.path.join(self._json_dir, json_data['imagePath'])
        img_h, img_w, _ = cv2.imread(img_path).shape

        for shape in json_data['shapes']:
            if shape['shape_type'] == 'polygon':
                yolo_obj = self._get_polygon_shape_yolo_object(shape, img_h, img_w)
            
            if yolo_obj:
                yolo_obj_list.append(yolo_obj)
        
        return yolo_obj_list

    def _get_polygon_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        
        # Start with the label ID
        yolo_object = [label_id]
        
        # Normalize each point of the polygon
        for point in shape['points']:
            normalized_x = round(float(point[0]) / img_w, 6)
            normalized_y = round(float(point[1]) / img_h, 6)
            yolo_object.extend([normalized_x, normalized_y])
        
        return yolo_object

    def _save_yolo_label(self, json_name, yolo_obj_list):
        # Save the YOLO label file in the same directory as the JSON file
        txt_path = os.path.join(self._json_dir, json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj in yolo_obj_list:
                # Join the coordinates into a single line, space-separated
                yolo_obj_line = " ".join(str(i) for i in yolo_obj)
                f.write(f'{yolo_obj_line}\n')

# Example usage:
# json_dir = "/path/to/your/labelme/json/files"
# converter = Labelme2YOLOSegmentation(json_dir)
# converter.convert()
