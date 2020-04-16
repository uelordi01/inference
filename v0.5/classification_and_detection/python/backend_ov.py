"""
pytoch/caffe2 backend via onnx
https://pytorch.org/docs/stable/onnx.html
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock
import os
# needed to get version and cuda setup
from openvino.inference_engine import IENetwork, IECore
import backend
import logging as log
import numpy as np
import time
class BackendOpenVino(backend.Backend):
    def __init__(self):
        super(BackendOpenVino, self).__init__()
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.lock = Lock()
        self.plugin = IECore()
    def version(self):
        return "no version defined:"

    def name(self):
        return "OpenVino"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        device="CPU"
        model_xml = model_path
        input_size = 1
        output_size = 1
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        cpu_extension = '/home/VICOMTECH/uelordi/SDK/MLPERF/mlperf_inference/v0.5/classification_and_detection/openvino_plugin/libcpu_extension.so'
        plugin = None
        # Plugin initialization for specified device
        # and load extensions library if specified
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_extension(cpu_extension, "CPU")
        if not device == 'HDDL':
            tag = {}
        # Read IR
        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the plugin...")

        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = \
                [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the plugin for specified device {}:\n {}".
                          format(device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")

        # Loads network read from IR to the plugin
        self.net_plugin = self.plugin.load_network(network=self.net,
                                                   device_name=device,
                                                   num_requests=4,
                                                   config=tag)
        self.input_blob = next(iter(self.net.inputs))
        if len(self.net.inputs.keys()) == 2:
            self.input_blob = "data"
        self.out_blob = next(iter(self.net.outputs))
        assert len(self.net.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.net.inputs))
        assert len(self.net.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.net.outputs))
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            for key in self.net.inputs:
                self.inputs.append(key)
        if outputs:
            self.outputs = outputs
        else:
            for key in self.net.outputs:
                self.outputs.append(key)
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
            for number, proposal in enumerate(dnn_tuple[0][0]):
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
        # yield to give
        feed_image_size = len(feed)
        input_key_found = self.check_input(feed)
        feed_length = len(feed[self.inputs[0]])
        result = None

        for i in range (0, feed_length):
            self.net_plugin.start_async(request_id=i, inputs={self.inputs[0]:feed[self.inputs[0]][i]})
        return_statement = []
        result_list = []
        for i in range(0, feed_length):
            self.net_plugin.requests[i].wait(-1)
            res = self.net_plugin.requests[i].outputs[self.outputs[0]]
            result_list.append(res)
        result = self.postProcessingMobileNET(result_list)
        #x = np.reshape(result, (1, 4)).T

        #return_statement.append(x)
        return result



#{self.inputs[0]:input_key}
    def check_input(self, feed):
        keys = feed.keys()
        input_key_found = False
        for key in keys:
            if key == self.inputs[0]:
                input_key_found = True
        return input_key_found
