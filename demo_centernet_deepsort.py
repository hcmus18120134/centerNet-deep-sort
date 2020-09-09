def warn(*args, **kwargs):
        pass
    
import warnings
warnings.warn = warn
import argparse
import os
#CenterNet
import sys
import time
CENTERNET_PATH = './CenterNet/src/lib'
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
import queue
import torch.multiprocessing as mp

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-i", "--input", help="Your input video path.")
    parser.add_argument("-o", "--output", help="Your destination output video path.")
    parser.add_argument("-cp", "--checkpoint", help="Your model path.")
    parser.add_argument("-t", "--text_output", help="Your text output path.")
    parser.add_argument("-w", "--write_vid", help="= True if write video.")
    parser.add_argument("-skip", "--skip_frame", default= 0, help="= True if write video.")
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
options.skip_frame = int(options.skip_frame)


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


class QueueItem():
    def __init__(self, detect_results, imgs, ori_imgs, frame_ids):
        self.detect_results = detect_results
        self.imgs = imgs
        self.ori_imgs = ori_imgs
        self.frame_ids = frame_ids
        
class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()

        #centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.detector.model.share_memory()
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
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)
            # self.vdo = [os.path.join(opt.vid_path,x) for x in os.listdir(opt.vid_path)]
            # self.vdo.sort()

        # self.im_height = int(sample.shape[0]) load folder of frames
        # self.im_width = int(sample.shape[1]) load folder of frames
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.area = 0, 0, self.im_width, self.im_height

        print(self.area)
        if is_write:
            if self.write_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output = cv2.VideoWriter(options.output, fourcc, 10, (self.im_width, self.im_height))

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    def detect(self):
        self.skipped_frames = 0
        xmin, ymin, xmax, ymax = self.area
        total_frames = range((13500))
        pbar = tqdm(total_frames)
        pbar.total = 13500
        pbar.refresh()
        frame_id = 0
        skip = True
        print('\n======================\n')
        print('skip {} frames mode'.format(options.skip_frame))
        print('\n======================\n')
        batch_size = 16

        batch_count = 0
        imgs = []
        ori_imgs = []
        frame_ids = []
        queue_items = mp.Queue(maxsize=5)
        proc_tracking = mp.Process(target=tracking, args=(queue_items, self.area))
        proc_tracking.start()
        time.sleep(2) # wait for load model deepsort in Tracking
        while self.vdo.grab():
            ret, ori_im = self.vdo.retrieve()
            if ret == False:
                print('End of video')
                break
            frame_id += 1
            if (frame_id %(options.skip_frame+2)==2): continue
            batch_count += 1
            
            im = ori_im[ymin:ymax, xmin:xmax]
            imgs.append(im)
            ori_imgs.append(ori_im)
            frame_ids.append(frame_id)

            if batch_count == batch_size:
                # print('Reach batch size!!')
                results = self.detector.run(imgs)['results']
                queue_item = QueueItem(results, imgs, ori_imgs, frame_ids)
                queue_items.put(queue_item, block=True)
                batch_count = 0
                imgs = []
                ori_imgs = []
                frame_ids = []

            if is_write:
                if self.write_video:
                    self.output.write(ori_im)
            
            pbar.update(1)
        if batch_count > 0:
            print('Process last batch')
            results = self.detector.run(imgs)['results']
            queue_item = QueueItem(results, imgs, ori_imgs, frame_ids)
            queue_items.put(queue_item, block=True)
            batch_size = 0
            imgs = []
            ori_imgs = []
            frame_ids = []
        pbar.close()
        proc_tracking.join()
        self.output.release()


def tracking(queue_items: mp.Queue, area):
    txt_writer = open(txt_path, 'wt')
    deepsorts = []
    for i in range(5):
        deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        deepsort.extractor.net.share_memory()
        deepsorts.append(deepsort)
    xmin, ymin, xmax, ymax = area
    while True:
        try:
            queue_item = queue_items.get(block=True, timeout=3)
        except queue.Empty:
            print('Empty queue. End?')
            break

        batch_results = queue_item.detect_results
        imgs = queue_item.imgs
        ori_imgs = queue_item.ori_imgs
        frame_ids = queue_item.frame_ids
        for batch_idx, results in enumerate(batch_results): # frame by frame
            for class_id in [1,2,3,4]:
                bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results,class_id)
                if (bbox_xywh is not None) and (len(bbox_xywh) > 0):
                    outputs = deepsorts[class_id].update(bbox_xywh, cls_conf, imgs[batch_idx])
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        
                        offset=(xmin, ymin)
                        if is_write:
                            ori_im = draw_bboxes(ori_imgs[batch_idx], bbox_xyxy, identities, class_id, offset=(xmin, ymin))
                        for i,box in enumerate(bbox_xyxy):
                            x1,y1,x2,y2 = [int(i) for i in box]
                            x1 += offset[0]
                            x2 += offset[0]
                            y1 += offset[1]
                            y2 += offset[1]
                            idx = int(identities[i]) if identities is not None else 0    
                            txt_writer.write(f'{frame_ids[batch_idx]} {class_id} {idx} {x1} {y1} {x2} {y2}\n')
    txt_writer.close()



if __name__ == "__main__":
    mp.set_start_method('spawn')
    import sys
    
    print(is_write)

    det = Detector(opt)

    det.open()
    det.detect()
