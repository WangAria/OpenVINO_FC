from openvino.inference_engine import IECore
import numpy as np
import time
import cv2 as cv
import os,sys 
from camvid.mapping import decode

import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for inference with models trained on CamVid data and optimized by OpenVINO')
    parser.add_argument('-d',
                        help='CPU|GPU',
                        default='cpu')
    
    return parser.parse_args(args)

def color_label(img, id2code):
    rows, cols = img.shape
    result = np.zeros((rows, cols, 3), 'uint8')
    for j in range(rows):
        for k in range(cols):
            result[j, k] = id2code[img[j, k]]
    return result

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    #1.Initialize inference engine core
    ie = IECore()
    for device in ie.available_devices:
        print(device)

    label_codes, label_names, code2id = decode('camvid-master/label_colors.txt')
    id2code = {val: key for (key, val) in code2id.items()}
    print(id2code)
    
    #2.Read a model in OpenVINO Intermediate Representation or ONNX format
    model_xml = "models/my_tiramisu.xml"
    model_bin = "models/my_tiramisu.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    #3.Configure input & output
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    #4.Loading model to the device
    exec_net = ie.load_network(network=net, device_name=args.d)
    
    #5.Prepare input
    n, h, w, c = net.input_info[input_blob].input_data.shape
    print(n, h, w, c)
    test = 'images/test_image1.png'
    frame = cv.imread(test)
    print(frame.shape)
    image = cv.resize(frame, (w, h))
    #image = image.transpose(2, 0, 1)
    print(image.shape)
    img_input = image[np.newaxis,:]

    #6.Create infer request, 7.Do inference
    infer_time_list = []
    inf_start = time.time()
    res = exec_net.infer(inputs={input_blob:img_input})
    inf_end = time.time() - inf_start
    infer_time_list.append(inf_end)

    #8.Process output
    res = res[out_blob].reshape((n, h, w, 32))
    res = np.squeeze(res, 0)
    res = np.argmax(res, axis=-1)
    hh, ww = res.shape
    print(res.shape)
    mask = color_label(res,id2code)    
    mask = cv.resize(mask, (frame.shape[1], frame.shape[0]))
    result = cv.addWeighted(frame, 0.5, mask, 0.5, 0)
    cv.putText(result, "infer time(ms): %.3f, FPS: %.2f"%(inf_end*1000, 1/(inf_end+0.0001)), (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
    cv.imshow("semantic segmentation benchmark", result)
    cv.waitKey(0) # wait for the image show

    infer_times = np.array(infer_time_list)
    avg_infer_time = np.mean(infer_times)
    print("infer time(ms): %.3f, FPS: %.2f"%(avg_infer_time*1000, 1/(avg_infer_time+0.0001)))
    cv.destroyAllWindows()

if __name__ == "__main__":
    
    main()