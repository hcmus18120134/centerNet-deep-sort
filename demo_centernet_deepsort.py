import argparse
import os
#CenterNet
import sys
import time
CENTERNET_PATH = '/content/centerNet-deep-sort/CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)
from tqdm import tqdm
import cv2
import numpy as np
# In order to solve CUDA OOM issue
import torch
torch.backends.cudnn.deterministic = True
from deep_sort import DeepSort
from opts import opts
from util import COLORS_10, draw_bboxes
from detectors.detector_factory import detector_factory



def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i", "--input", help="Your input video path.")
    parser.add_argument("-o", "--output", help="Your destination output video path.")
    parser.add_argument("-cp", "--checkpoint", help="Your model path.")
    parser.add_argument("-t", "--text_output", help="Your text output path.")
    parser.add_argument("-w", "--write_vid", help="= True if write video.")
    options = parser.parse_args(args)
    return options

options = getOptions(sys.argv[1:])
print(options)
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
MODEL_PATH = options.checkpoint
ARCH = 'dla_34'
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))

txt_path = options.text_output
opt.vid_path = options.input  
is_write = options.write_vid



#vis_thresh
opt.vis_thresh = 0.2


#input_type
opt.input_type = 'vid'   # for video, 'vid',  for webcam, 'webcam', for ip camera, 'ipcam'

#------------------------------
# for video
#------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
#------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
#------------------------------





def bbox_to_xywh_cls_conf(bbox,class_id):
    #confidence = 0.5
    # only person
    bbox = bbox[class_id]
    # print(bbox[:5, :])

    if any(bbox[:, 4] > opt.vis_thresh):

        bbox = bbox[bbox[:, 4] > opt.vis_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #

        return bbox[:, :4], bbox[:, 4]

    else:

        return None, None


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()


        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = [] 
        for i in range(5):
            self.deepsort.append(DeepSort("deep/checkpoint/ckpt.t7"))


        self.write_video = True

    def open(self):

        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))

        # video
        else :
            # assert os.path.isfile(opt.vid_path), "Error: path error"
            # self.vdo.open(opt.vid_path)
            self.vdo = [os.path.join(opt.vid_path,x) for x in os.listdir(opt.vid_path)]
            self.vdo.sort()
        sample = cv2.imread(self.vdo[0])
        self.im_height = int(sample.shape[0])
        self.im_width = int(sample.shape[1])

        self.area = 0, 0, self.im_width, self.im_height
        if is_write:
            if self.write_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output = cv2.VideoWriter(options.output, fourcc, 10, (self.im_width, self.im_height))



    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        try: 
          os.makedirs(txt_path)
        except: 
            pass

        total_frames = range(len(self.vdo))
        pbar = tqdm(self.vdo)
        # while self.vdo.grab():
        for path in pbar:
            ori_im = cv2.imread(path)
            txt_file = os.path.join(txt_path,'{:05}.txt'.format(frame_no))
            f = open(txt_file,'w')
            start = time.time()
            im = ori_im[ymin:ymax, xmin:xmax]

            results = self.detector.run(im)['results']
            for class_id in [1,2,3,4]:
                bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results,class_id)

                if bbox_xywh is not None:
                    outputs = self.deepsort[class_id].update(bbox_xywh, cls_conf, im)

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        
                        offset=(xmin, ymin)
                        if is_write:
                            ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, class_id, offset=(xmin, ymin))
                        for i,box in enumerate(bbox_xyxy):
                            x1,y1,x2,y2 = [int(i) for i in box]
                            x1 += offset[0]
                            x2 += offset[0]
                            y1 += offset[1]
                            y2 += offset[1]
                            idx = int(identities[i]) if identities is not None else 0    
                            f.write(f'{frame_no} {idx} {class_id} {x1} {y1} {x2} {y2}\n')

            end = time.time()

            fps =  1 / (end - start )

            avg_fps += fps

            if is_write:
                if self.write_video:
                    self.output.write(ori_im)
            f.close()

            frame_no +=1
            pbar.set_description("frame_id: {} fps: {:.2f}, avg fps : {:.2f}".format(frame_no, fps,  avg_fps/frame_no))
        pbar.close()
        self.output.release()


if __name__ == "__main__":
    import sys
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    print(is_write)

    det = Detector(opt)

    det.open()
    det.detect()
