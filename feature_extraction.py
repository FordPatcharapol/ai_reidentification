import cv2
import yaml
from turbojpeg import TurboJPEG
from openvino.inference_engine import IECore
import urllib.request
import json

# ***** read config yaml *****
with open("config.yaml", "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = config['model']
device = model['device']
model_name = model['name']
path_model = model['path']
type_model = model['type']

# ***** define value *****
ref_path = path_model + model_name + '/'+ type_model + '/' + model_name
ie = IECore()
classification_model_xml = ref_path + '.xml'
classification_model_bin = ref_path + '.bin'
net = ie.read_network(model=classification_model_xml, weights=classification_model_bin)
exec_net = ie.load_network(network=net, device_name=device, num_requests=2)
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
input_dims = net.input_info[input_blob].input_data.shape
output_dims = net.outputs[out_blob].shape
n, c, h, w = input_dims

jpeg = TurboJPEG()

def extract_feature_image(object_image):
    exec_net.infer(inputs={input_blob: object_image})
    res = exec_net.requests[0].output_blobs[out_blob].buffer
    feature_vec = res.reshape(1, 256)

    return feature_vec.tolist()

def sub_message_from_kafka():
    message = {
        "images" : [
            {
                "image": {
                    # *** mock data url ***
                    "url": './data/road.jpg',
                    "type": 'memcached',
                    'server': '172.30.236.126:11201',
                    'blobId': 'rpca_mobile_1669353683.295849',
                    'expireAt': '2022-11-25T05:21:33.417Z'
                },
                "boxes": {
                    '1669353640.26073-nvprhf': [996, 342, 1062, 516],
                    '1669353647.560987-8i93yx': [1014, 397, 1087, 582],
                    '1669353647.560987-c9d0oy': [1048, 302, 1112, 463]
                }
            }
        ]
    }
    return message

def reply_message_to_kafka(message):
    print('reply')

def show_image_window(image):
    while(1):
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def compose_message(feature_each_boxes, boxes_in_frame, boxes_id, frame):
    message = {
        "images" : []
    }

    boxes = {
        "boxes": {}
    }
    

    object_boxes = frame['boxes']
    index_id = 0
    
    for feature in feature_each_boxes:
        box_id = boxes_id[index_id]
        each_object = {
            'image': {},
            'box': [],
            'representation': {
                'model': 'openvino_0031',
                'dimension': 0,
                'vector': []
            }
        }
        each_object['image'] = frame['image']
        each_object['box'] = object_boxes[box_id]
        each_object['representation']['model'] = model_name
        each_object['representation']['dimension'] = len(feature)
        each_object['representation']['vector'] = feature
        index_id = index_id + 1
        boxes['boxes'][box_id] = each_object
    message['images'].append(boxes)
    return message

def load_image(img_path, image_type):

    if image_type == 'url':
        url_response = urllib.request.urlopen(img_path)
        image = jpeg.decode(bytearray(url_response.read()))
    elif image_type == 'base64':
        image = cv2.imread(img_path)
    elif image_type == 'file':
        in_file = open(img_path, 'rb')
        image = jpeg.decode(in_file.read())
        in_file.close()
    elif image_type == 'memcached':
        in_file = open(img_path, 'rb')
        image = jpeg.decode(in_file.read())
        in_file.close()

    else:
        return None

    return image

def write_to_json(messages):
    with open('./feature_message.json', "w") as outfile:
        json.dump(messages, outfile)

def prepare_image(object_box, image_base):
    x1, y1, x2, y2 = object_box
    im = image_base[y1: y2, x1: x2]
    # print(image_base.shape)
    # show_image_window(im)

    resized_image = cv2.resize(im, (w, h))
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = resized_image.reshape((n, c, h, w))
    # print(resized_image.shape)

    return resized_image

def main():
    # loop 
    messages = sub_message_from_kafka()
    frames = messages['images']
    for frame in frames:
        type_img = frame['image']['type']
        path_img = frame['image']['url']
        image_base = load_image(path_img, type_img)
        feature_each_boxes = []
        if image_base is None:
            continue
        boxes_in_frame = frame['boxes']
        boxes_id = []
        
        for object_info in boxes_in_frame:
            boxes_id.append(object_info)
            object_box = boxes_in_frame[object_info]
            object_prepared = prepare_image(object_box ,image_base)
            feature = extract_feature_image(object_prepared)
            feature_each_boxes.append(feature[0])            

        message_result = compose_message(feature_each_boxes, boxes_in_frame, boxes_id, frame)
        reply_message_to_kafka(message_result)
if __name__ == "__main__":
    main()