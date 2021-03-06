"""
pytoch/caffe2 backend via onnx
https://pytorch.org/docs/stable/onnx.html
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock
import os
# needed to get version and cuda setup
import cv2 as dldt
import backend
import logging as log
import numpy as np
import time
class BackendDldt(backend.Backend):
    def __init__(self):
        super(BackendDldt, self).__init__()
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.det_net = None
        self.__out_names = None
        self.__image_src = None
    def version(self):
        return "no version defined:"

    def name(self):
        return "OpenCV dldt"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        det_config_file=model_path
        det_model_file = os.path.splitext(model_path)[0] + ".bin"
        self.det_net = dldt.dnn.readNet(det_config_file, det_model_file)
        self.det_net.setPreferableBackend(dldt.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.det_net.setPreferableTarget(dldt.dnn.DNN_TARGET_CPU)
        if self.det_net == None:
            print("error the model is not loaded")
        self.__out_names = self.det_net.getUnconnectedOutLayersNames()
        if inputs:
           self.inputs = inputs
        else:
            print("inputs have to be defined in the script parameter")
        #     self.inputs = []
        #     for key in self.net.inputs:
        #         self.inputs.append(key)
        layerNames = self.det_net.getLayerNames()
        print("loading dldt finished")
        self.outputs = layerNames
        if outputs:
            self.outputs = outputs
        else:
            for key in self.net.outputs:
                self.outputs.append(key)

        # TODO define the inputs correclty pelase
        return self
    def postProcessingMobileNET(self, res):
        #take only the 95% values
        detection_classes = []
        detection_boxes_list = []
        detection_num_list = []
        detection_classes_list = []
        detection_threshold_list = []
        length_res = len(res[0][0])
        for dnn_tuple in res:
            detection_boxes = []
            detection_num = 0
            detection_classes = []
            detection_threshold = []
            for number, proposal in enumerate(dnn_tuple[0][0][0]):
                if proposal[2] > 0:
                    detection_classes.append(np.int(proposal[1]))
                    detection_num = detection_num + 1
                    detection_threshold.append(proposal[2])
                    detection_boxes.append(np.array([proposal[3],
                                                     proposal[4],
                                                     proposal[5],
                                                     proposal[6]]))
            detection_boxes_list.append(detection_boxes)
            detection_num_list.append(detection_num)
            detection_threshold_list.append(detection_threshold)
            detection_classes_list.append(detection_classes)

        # data = res[0][0][0]
        # for number, proposal in enumerate(data):
        #     if proposal[2] > 0:
        #         detection_classes.append(np.int(proposal[1]))
        #         detection_num = detection_num + 1
        #         detection_threshold.append(proposal[2])
        #         detection_boxes.append(np.array([proposal[3], proposal[4], proposal[5], proposal[6]]))
        return [detection_num_list,
                detection_boxes_list,
                detection_threshold_list,
                detection_classes_list]


    def predict(self, feed):
        # TODO put the input correctly
        feed_length = len(feed[self.inputs[0]])
        result_list = []

        for i in range(0, feed_length):
             input_shape = None
             input_shape = feed[self.inputs[0]][i].reshape(1, 3, 300, 300)
             self.det_net.setInput(input_shape, self.inputs[0])
             outs = self.det_net.forward(self.__out_names)
             result_list.append(outs)
        result = self.postProcessingMobileNET(result_list)
        return result



#{self.inputs[0]:input_key}
    def check_input(self, feed):
        keys = feed.keys()
        input_key_found = False
        for key in keys:
            if key == self.inputs[0]:
                input_key_found = True
        return input_key_found
