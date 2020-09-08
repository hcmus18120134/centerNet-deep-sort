def warn(*args, **kwargs):
        pass
    
import warnings
warnings.warn = warn
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
    def __init__(self, detect_results, imgs, ori_imgs, frame_ids, end_flag=False):
        self.detect_results = detect_results
        self.imgs = imgs
        self.ori_imgs = ori_imgs
        self.frame_ids = frame_ids
        self.end_flag = end_flag
        
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
        print('skip {} frames mode'.format(options.skip_frame))
        batch_size = 4


        class_ids = [1,2,3,4]
        input_queues = [mp.Queue() for _ in range(5)]
        output_queue = mp.Queue()
        ready_flags = [mp.Event() for _ in range(5)]
        flag_start = mp.Event()
        flag_start.clear()
        track_one_class_procs = [
            mp.Process(target=handle_track_one_class, args=(xmin, ymin, input_queues[class_id], flag_start, output_queue, ready_flags[class_id]))
            for class_id in class_ids
        ]
        proc_write_result = mp.Process(target=handle_write_result, args=(txt_path, output_queue))
        for proc in track_one_class_procs:
            proc.start()

        print('Waiting for loading weights...')
        for flag in ready_flags[1:]: # flag with class_id = 0 will never be set
            if not flag.is_set():
                flag.wait()

        batch_count = 0
        imgs = []
        ori_imgs = []
        frame_ids = []
        queue_items = mp.Queue(maxsize=5)
        proc_tracking = mp.Process(target=tracking, args=(queue_items, input_queues, class_ids, flag_start))
        proc_tracking.start()

        print('Start!!')
        flag_start.set()
        # time.sleep(3) # wait for load model deepsort in Tracking
        proc_write_result.start()
        while self.vdo.grab():
            ret, ori_im = self.vdo.retrieve()
            if ret == False:
                print('End of video')
                break
            frame_id += 1
            batch_count += 1
            
            im = ori_im[ymin:ymax, xmin:xmax]
            imgs.append(im)
            ori_imgs.append(ori_im)
            frame_ids.append(frame_id)

            if batch_count == batch_size:
                # print('Reach batch size!!')
                results = self.detector.run(imgs)['results']
                queue_item = QueueItem(results, imgs, ori_imgs, frame_ids, False)
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
            # print('Process last batch')
            results = self.detector.run(imgs)['results']
            queue_item = QueueItem(results, imgs, ori_imgs, frame_ids, False)
            queue_items.put(queue_item, block=True)
            batch_size = 0
            imgs = []
            ori_imgs = []
            frame_ids = []

        # Send end signals
        queue_item = QueueItem(None, None, None, None, True)
        queue_items.put(queue_item, block=True)

        pbar.close()
        proc_tracking.join()
        for proc in track_one_class_procs:
            proc.join()
        proc_write_result.join()
        self.output.release()


def tracking(queue_items: mp.Queue, input_queues, class_ids, flag_start: mp.Event):

    flag_start.wait()

    while True:
        try:
            # print('Get tracking queue_item')
            queue_item = queue_items.get(block=True)
            # print('Got tracking queue_item')
        except queue.Empty:
            print('Empty Tracking Queue. End?')
            break

        if queue_item.end_flag:
            input_ = TrackOneClassInput(None, None, None, None, None, True)
            for q in input_queues:
                q.put(input_)
            break

        batch_results = queue_item.detect_results
        imgs = queue_item.imgs
        ori_imgs = queue_item.ori_imgs
        frame_ids = queue_item.frame_ids
        for batch_idx, results in enumerate(batch_results): # frame by frame
            frame_id = frame_ids[batch_idx]
            frame_image = imgs[batch_idx]
            for class_id in class_ids:
                bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results, class_id)
                if bbox_xywh is not None:
                    # print('Feed TrackOneClassInput')
                    inputs = TrackOneClassInput(class_id, bbox_xywh, cls_conf, frame_id, frame_image, False)
                    input_queues[class_id].put(inputs)

def handle_write_result(txt_path, input_queue):
    txt_writer = open(txt_path, 'wt')

    while True:
        try:
            inputs = input_queue.get(block=True)
        except queue.Empty:
            print('Empty Writing Queue. End!')
            break

        if inputs.end_flag:
            break

        line = inputs.line
        txt_writer.write(line)

    txt_writer.close()

def handle_track_one_class(xmin, ymin, input_queue: mp.Queue, input_flag_start: mp.Event, output_queue: mp.Queue, output_flag_ready: mp.Event):
    deepsort = DeepSort("deep/checkpoint/ckpt.t7")
    deepsort.extractor.net.share_memory()
    output_flag_ready.set()

    input_flag_start.wait()

    while True:
        try:
            inputs = input_queue.get(block=True)
            # print('Got inputs handle_track_one_class')
        except queue.Empty:
            print('Deepsort Empty Queue. End!')
            break

        if inputs.end_flag:
            break

        class_id = inputs.class_id
        frame_image = inputs.frame_image
        frame_id = inputs.frame_id
        bbox_xywh = inputs.bbox_xywh
        cls_conf = inputs.cls_conf

        outputs = deepsort.update(bbox_xywh, cls_conf, frame_image)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            
            offset=(xmin, ymin)
            # if is_write:
            #     ori_im = draw_bboxes(ori_imgs[batch_idx], bbox_xyxy, identities, class_id, offset=(xmin, ymin))
            for box_idx, box in enumerate(bbox_xyxy):
                x1,y1,x2,y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]
                idx = int(identities[box_idx]) if identities is not None else 0
                line = f'{frame_id} {class_id} {idx} {x1} {y1} {x2} {y2}\n'
                output_queue.put(TrackOneClassOutput(line))


class TrackOneClassInput:
    def __init__(self, class_id, bbox_xywh, cls_conf, frame_id, frame_image, end_flag=False):
        self.class_id = class_id
        self.bbox_xywh = bbox_xywh
        self.cls_conf = cls_conf
        self.frame_id = frame_id
        self.frame_image = frame_image
        self.end_flag = end_flag

class TrackOneClassOutput:
    def __init__(self, line, end_flag=False):
        self.line = line
        self.end_flag = end_flag

if __name__ == "__main__":
    mp.set_start_method('spawn')
    import sys
    def warn(*args, **kwargs):
        pass
    
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.warn = warn
    print(is_write)

    det = Detector(opt)

    det.open()
    det.detect()
