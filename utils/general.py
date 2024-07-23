import os
import cv2
import json
import math
import parse
import shutil
import numpy as np
import pandas as pd

from collections import deque
from PIL import Image, ImageDraw
from model import TrackNet, InpaintNet

# from .func_clips_start_end import *

from ultralytics import YOLO
import cv2
import numpy as np




# Global variables
HEIGHT = 288
WIDTH = 512
SIGMA = 2.5
DELTA_T = 1/math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH = DELTA_T * 50
IMG_FORMAT = 'png'

class ResumeArgumentParser():
    """ A argument parser for parsing the parameter dictionary from checkpoint file."""
    def __init__(self, param_dict):
        self.model_name = param_dict['model_name']
        self.seq_len = param_dict['seq_len']
        self.epochs = param_dict['epochs']
        self.batch_size = param_dict['batch_size']
        self.optim = param_dict['optim']
        self.learning_rate = param_dict['learning_rate']
        self.lr_scheduler = param_dict['lr_scheduler']
        self.bg_mode = param_dict['bg_mode']
        self.alpha = param_dict['alpha']
        self.frame_alpha = param_dict['frame_alpha']
        self.mask_ratio = param_dict['mask_ratio']
        self.tolerance = param_dict['tolerance']
        self.resume_training = param_dict['resume_training']
        self.seed = param_dict['seed']
        self.save_dir = param_dict['save_dir']
        self.debug = param_dict['debug']
        self.verbose = param_dict['verbose']


###################################  Helper Functions ###################################
def get_model(model_name, seq_len=None, bg_mode=None):
    """ Create model by name and the configuration parameter.

        Args:
            model_name (str): type of model to create
                Choices:
                    - 'TrackNet': Return TrackNet model
                    - 'InpaintNet': Return InpaintNet model
            seq_len (int, optional): Length of input sequence of TrackNet
            bg_mode (str, optional): Background mode of TrackNet
                Choices:
                    - '': Return TrackNet with L x 3 input channels (RGB)
                    - 'subtract': Return TrackNet with L x 1 input channel (Difference frame)
                    - 'subtract_concat': Return TrackNet with L x 4 input channels (RGB + Difference frame)
                    - 'concat': Return TrackNet with (L+1) x 3 input channels (RGB)

        Returns:
            model (torch.nn.Module): Model with specified configuration
    """

    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            model = TrackNet(in_dim=seq_len*4, out_dim=seq_len)
        elif bg_mode == 'concat':
            model = TrackNet(in_dim=(seq_len+1)*3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len*3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name.')
    
    return model

def list_dirs(directory):
    """ Extension of os.listdir which return the directory pathes including input directory.

        Args:
            directory (str): Directory path

        Returns:
            (List[str]): Directory pathes with pathes including input directory
    """

    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])

def to_img(image):
    """ Convert the normalized image back to image format.

        Args:
            image (numpy.ndarray): Images with range in [0, 1]

        Returns:
            image (numpy.ndarray): Images with range in [0, 255]
    """

    image = image * 255
    image = image.astype('uint8')
    return image

def to_img_format(input, num_ch=1):
    """ Helper function for transforming model input sequence format to image sequence format.

        Args:
            input (numpy.ndarray): model input with shape (N, L*C, H, W)
            num_ch (int): Number of channels of each frame.

        Returns:
            (numpy.ndarray): Image sequences with shape (N, L, H, W) or (N, L, H, W, 3)
    """

    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    
    if num_ch == 1:
        # (N, L, H ,W)
        return input
    else:
        # (N, L*C, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, L*C)
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3) # (N, L, H, W, 3)
        # For each sample in the batch
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            # Get each frame in the sequence
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq

def get_num_frames(rally_dir):
    """ Return the number of frames in the video.

        Args:
            rally_dir (str): File path of the rally frame directory 
                Format: '{data_dir}/{split}/match{match_id}/frame/{rally_id}'

        Returns:
            (int): Number of frames in the rally frame directory
    """

    try:
        frame_files = list_dirs(rally_dir)
    except:
        raise ValueError(f'{rally_dir} does not exist.')
    frame_files = [f for f in frame_files if f.split('.')[-1] == IMG_FORMAT]
    return len(frame_files)

def get_rally_dirs(data_dir, split):
    """ Return all rally directories in the split.

        Args:
            data_dir (str): File path of the data root directory
            split (str): Split name

        Returns:
            rally_dirs: (List[str]): Rally directories in the split
                Format: ['{split}/match{match_id}/frame/{rally_id}', ...]
    """
    rally_dirs = []

    # Get all match directories in the split
    match_dirs = os.listdir(os.path.join(data_dir, split))
    match_dirs = [os.path.join(split, d) for d in match_dirs]
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split('match')[-1]))
    
    # Get all rally directories in the match directory
    for match_dir in match_dirs:
        rally_dir = os.listdir(os.path.join(data_dir, match_dir, 'frame'))
        rally_dir = sorted(rally_dir)
        rally_dir = [os.path.join(match_dir, 'frame', d) for d in rally_dir]
        rally_dirs.extend(rally_dir)
    
    return rally_dirs


def generate_frames_from_cap(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = []
    success = True

    # Sample frames until video end
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)
            
    return frame_list, fps, (w, h)

def generate_frames(video_file):
    """ Sample frames from the video.

        Arlgs:
            video_file (str): File path of the video file

        Returns:
            frame_list (List[numpy.ndarray]): List of sampled frames
            fps (int): Frame per second of the video
            (w, h) (Tuple[int, int]): Width and height of the video
    """

    assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    # Get camera parameters
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = []
    success = True

    # Sample frames until video end
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)
            
    return frame_list, fps, (w, h)

def draw_traj(img, traj, radius=3, color='red'):
    """ Draw trajectory on the image.

        Args:
            img (numpy.ndarray): Image with shape (H, W, C)
            traj (deque): Trajectory to draw

        Returns:
            img (numpy.ndarray): Image with trajectory drawn
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = Image.fromarray(img)
    
    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox =  (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img

def write_pred_video(frame_list, video_cofig, pred_dict, save_file, traj_len=8, label_df=None):
    """ Write a video with prediction result.

        Args:
            frame_list (List[numpy.ndarray]): List of sampled frames
            video_cofig (Dict): Video configuration
                Format: {'fps': fps (int), 'shape': (w, h) (Tuple[int, int])}
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int])}
            save_file (str): File path of the output video file
            traj_len (int, optional): Length of trajectory to draw
            label_df (pandas.DataFrame, optional): Ground truth label dataframe
        
        Returns:
            None
    """

    # Read ground truth label if exists
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    
    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, video_cofig['fps'], video_cofig['shape'])
    
    # Create a queue for storing trajectory
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()
    
    # Draw label and prediction trajectory
    for i, frame in enumerate(frame_list):
        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()
        
        # Push ball coordinates for each frame
        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # Draw ground truth trajectory if exists
        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        
        # Draw prediction trajectory
        frame = draw_traj(frame, pred_queue, color='yellow')
        # print(i)
        out.write(frame)
    out.release()



def pred_dict_modify(args, pred_dict, frame_list, video_config):
    x_pred = pred_dict['X']
    y_pred = pred_dict['Y']
    vis_pred = pred_dict['Visibility']
    
    add_frame = []
    wait_counter = -1
    disappearance = -1
    disappearance_reason = 0

    ''' FIRST ROUND '''
    for i, frame in enumerate(frame_list):

        skip_frame = 0
        if (disappearance != 1):

            # Height check. disappearance here simply denotes whether the shuttle disappeared because it was too high and out of frame. 0 - cut the video because of disappearance, 1 - do not cut the video, as it disappeared because of being too high
            if (vis_pred[i] == 0) and (i > 0) and (vis_pred[i-1] == 1 and y_pred[i-1] > 0 and y_pred[i-1] < video_config['shape'][1]*0.2):
                disappearance = 1

            elif (vis_pred[i] == 0) and (i == 0):
                disappearance = 1

        # shuttle is visible
        if (disappearance == 1) and (vis_pred[i] == 1):
            disappearance = -1
            wait_counter = -1

        ''' 
        check for stationary shuttle begins here
        '''

        if (i > 5) and disappearance != 1:
            counter = 0
            for k in range(15,0,-1):
                if not (((x_pred[i] - 6) <= (x_pred[i-k]) <= (x_pred[i] + 6)) and ((y_pred[i] - 6) <= (y_pred[i-k]) <= (y_pred[i] + 6))) or vis_pred[i-k] == 0:

                    counter += 1
            
            if (counter <= 10):
                disappearance_reason = 1
                skip_frame = 1

        ''''''

        # if shuttle doesn't reappear after disappearance due to height - due to non detection, just cut it out anyways
        
        if (disappearance == 1):
            flag = 1
            for k in range(75,0,-1):
                if (vis_pred[i-k] != 0):
                    flag = 0
            
            if (flag == 1):
                disappearance = 0
                skip_frame = 1


        if (disappearance != 1):
            if (vis_pred[i] == 0):
                disappearance = 0


        # if the shuttle goes missing but reappears within 8 frames        
        if (wait_counter >= 0) and (vis_pred[i] == 1):
            wait_counter = -1
            disappearance = -1

        if (wait_counter <= 0) and (skip_frame != 1):
            if (disappearance == 0) and (disappearance_reason != 1):
                disappearance = -1
                disappearance_reason = 0
                skip_frame = 1

            if (disappearance == 0) and (disappearance_reason == 1):
                skip_frame = 1
                wait_counter = 0

        if (skip_frame != 1):
            disappearance_reason = 0
            add_frame.append(True)
        else:
            add_frame.append(False)

        # print(i, skip_frame, vis_pred[i], wait_counter, disappearance, disappearance_reason)


    # for i, f in enumerate(add_frame):
    #     print(i,f)


    ''' INITIAL SMOOTHENING '''

    kernel = np.array([1.5,1.25,1,1,1,1,1,1,1.25,1.5], dtype=float)

    # Perform convolution with padding to avoid boundary effects.
    smoothed_data = np.convolve(add_frame, kernel, mode='same')

    # A value greater than or equal to the window size indicates a patch of 1s.
    add_frame = smoothed_data.tolist()

    for i,e in enumerate(add_frame):
        if e <= 5:
            add_frame[i] = False
        else:
            add_frame[i] = True
        # print(i,add_frame[i])


    ''' ROUND 1.5 '''
    for i in range(len(frame_list)-1, -1, -1):
        if (i > 0):
            if (add_frame[i] == False and add_frame[i-1] == True):

                for k in range(30):
                    try:
                        add_frame[i+k] = True
                    except:
                        continue     


    ''' SECOND ROUND '''
    for i, frame in enumerate(frame_list):

        # eliminate random detections from between
        
        if (i < len(frame_list) - 70):

            if (add_frame[i] == False and add_frame[i+1] == True):
                flag = 0
                for k in range(70):
                    if (add_frame[i+k] == True):
                        flag += 1
            

                if (flag <= 60):
                    for k in range(70):
                        add_frame[i+k] = False
        
        else:
            try:
                if (add_frame[i] == False and add_frame[i+1] == True and add_frame[-1] == False):
                    flag = 0
                    total_len = len(add_frame[i:])

                    for k in add_frame[i:]:
                        if (k == True):
                            flag += 1
                    
                    if (flag <= total_len*0.9):
                        add_frame[i:] = [False]*(total_len)


            except:
                pass



    ''' THIRD ROUND - padding before and after clips '''
    
    for i, frame in enumerate(frame_list):
        if (i < len(frame_list) - 1):
            if (add_frame[i] == False and add_frame[i+1] == True):

                for k in range(30):
                    if ((i - k) >= 0):
                        add_frame[i-k] = True
                    else:
                        if args:
                            try:
                                args[1][i-k] = True
                            except:
                                continue

    # for i,e in enumerate(add_frame):
    #     print(i,e)

    ''' FOURTH ROUND - For older clip '''

    # if args:
    #     add_frame_old = args[1]
    # else:
    #     add_frame_old = None

    # flag = -100


    # if (add_frame_old):
    #     for i in range(-1,-100,-1):
    #         if (add_frame_old[i] == False):
    #             flag = i
    #             break

    #         if (add_frame_old[i] == True and add_frame_old[i-1] == False):
    #             flag = i
    #             # print("prev clip", i)
    #             break
        
    #     if (flag != -100):
    #         finished = False
    #         removal_array = add_frame_old[flag:]
    #     else:
    #         finished = True
    #         removal_array = []
    
    # else:
    #     removal_array = []
    

    # flag1 = 100

    # for i in range(100):
    #     if i < (len(add_frame) - 1):
    #         if (add_frame[i] == False):
    #             flag1 = i
    #             break

    #         if (add_frame[i] == True and add_frame[i+1] == False):
    #             flag1 = i
    #             # print("cur clip", i)
    #             break

    # print("flag1", flag1, "flag", flag)

    # if (flag1 != 100):
    #     finished_1 = False
    #     removal_array = removal_array + add_frame[:(i+1)]
    
    # else:
    #     finished_1 = True

    # print(len(removal_array), "REMOVAL ARRAY")
        
    # check_true = len(removal_array)
    # # print(check_true,"check_true")

    # if check_true <= 90 and check_true > 0:
    #     if (add_frame_old) and (flag != -100) and finished_1 == False:
    #         add_frame_old[flag:] = [False]*(flag * -1)
        
    #     if (flag1 != 100) and add_frame_old and finished == False:
    #         # if (add_frame_old[-1] == False):
    #         add_frame[:flag1 + 1] = [False]*(flag1 + 1)
        
    #     if (flag1 != 100) and (flag != -100):
    #         print("here")
    #         add_frame_old[flag:] = [False]*(flag * -1)
    #         add_frame[:flag1 + 1] = [False]*(flag1 + 1)

    # if (flag1 != 100) and not add_frame_old:
    #     if check_true <= 60:
    #         add_frame[:flag1 + 1] = [False] * (flag1 + 1)


    # if (args):
    #     for i,e in enumerate(args[1]):
    #         print(i,e)
 
    
    # elif (flag1 != 100) and add_frame_old:
    #     if check_true <= 60:
    #         add_frame[:flag1 + 1] = [False] * (flag1 + 1)

    

    return add_frame





def last_vid_pd_modify(add_frame):
    flag = -100

    if (add_frame):


        for i in range(-1,-100,-1):

            try:
                if (add_frame[i] == False):
                    break

                if (add_frame[i] == True and add_frame[i-1] == False):
                    flag = i
                    break
            except:
                return [False]*(len(add_frame))

        removal_array = add_frame[flag:]

    
    else:
        removal_array = []

    check_true = len(removal_array)
        
    if check_true <= 90:
        if (add_frame) and (flag != -100):
            add_frame[flag:] = [False]*(flag * -1)


    return add_frame
    


def write_pred_video_modified(frame_list, video_cofig, pred_dict, save_file, prev_last_frame, add_frame, traj_len=8, label_df=None):
    # print(save_file[11:])
    """ Write a video with prediction result.

        Args:
            frame_list (List[numpy.ndarray]): List of sampled frames
            video_cofig (Dict): Video configuration
                Format: {'fps': fps (int), 'shape': (w, h) (Tuple[int, int])}
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int])}
            save_file (str): File path of the output video file
            traj_len (int, optional): Length of trajectory to draw
            label_df (pandas.DataFrame, optional): Ground truth label dataframe
        
        Returns:
            None
    """
    disappearance = -1 # used later to check if shuttle went missing because of beFing too high

    # for i in range(len(add_frame)):
    #     print(i,add_frame[i],sep=",", end = " ")

    # Read ground truth label if exists
    if label_df is not None:
        f_i, x, y, vis = label_df['Frame'], label_df['X'], label_df['Y'], label_df['Visibility']
    
    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, video_cofig['fps'], video_cofig['shape'])
    # print(video_cofig['shape'])
    # Create a queue for storing trajectory
    pred_queue = deque()
    if label_df is not None:
        gt_queue = deque()
    # used it to create a smoother highlight
    wait_counter = -1
    reappearance_list = []
    
    # black_frame = np.zeros((video_cofig['shape'][1], video_cofig['shape'][0], 3), dtype=np.uint8)

    if (prev_last_frame == None or prev_last_frame == False):
        last_frame = False
    elif prev_last_frame == True:
        last_frame = True

    active_frame = 0
    active_frame_list = []
    frame_in_csv = []
    active_frame_returned = None
    file_name = save_file

    # Draw label and prediction trajectory
    for i, frame in enumerate(frame_list):

        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()
        if label_df is not None and len(gt_queue) >= traj_len:
            gt_queue.pop()
        
        # Push ball coordinates for each frame (Not relevant)
        if label_df is not None:
            gt_queue.appendleft([x[i], y[i]]) if vis[i] and i < len(label_df) else gt_queue.appendleft(None)


        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # Draw ground truth trajectory if exists (Not relevant)
        if label_df is not None:
            frame = draw_traj(frame, gt_queue, color='red')
        

        # if (skip_frame != 1):
        # print(i)

        if add_frame[i] == True:
            active_frame+=1 
            # print(i)
            frame = draw_traj(frame, pred_queue, color='yellow')
            out.write(frame)
        
        if (last_frame == True and add_frame[i] == False):
            # clip_end(i)
            print(i,"Ending clip")
            # out.write(blackframe)

        if (last_frame == False and add_frame[i] == True):
            print(i,"Starting clip")

            # clip_start(i, active_frame, save_file)
            frame_in_csv.append(i)
            active_frame_list.append(active_frame)
            # active_frame_returned = active_frame
            file_name = save_file
            # print("Finished")
            # out.write(blackframe)

        last_frame = add_frame[i]
        # if (skip_frame == 1 and add_to_list == True):
        #     reappearance_list.append(frame)
    
    out.release()
    return add_frame, frame_in_csv, active_frame_list, file_name



def write_pred_csv(pred_dict, save_file, save_inpaint_mask=False):

    """ Write prediction result to csv file.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int]),
                         'Inpaint_Mask': inpaint_mask (List[int])}
            save_file (str): File path of the output csv file
            save_inpaint_mask (bool, optional): Whether to save inpaint mask

        Returns:
            None
    """

    if save_inpaint_mask:
        # Save temporary data for InpaintNet training
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility_GT': pred_dict['Visibility_GT'],
                                'X_GT': pred_dict['X_GT'],
                                'Y_GT': pred_dict['Y_GT'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'], 
                                'Y': pred_dict['Y'],
                                'Inpaint_Mask': pred_dict['Inpaint_Mask']})
    else:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'],
                                'Y': pred_dict['Y']})
    pred_df.to_csv(save_file, index=False)
    
def convert_gt_to_coco_json(data_dir, split, drop=False):
    """ Convert ground truth csv file to coco format json file.

        Args:
            split (str): Split name
        
        Returns:
            None
    """
    if split == 'test' and drop:
        drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
        start_frame, end_frame = drop_frame_dict['start'], drop_frame_dict['end']
    bbox_size = 10
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    image_info = []
    annotations = []
    sample_count = 0
    for rally_dir in rally_dirs:
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]
        csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if split == 'test' else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        label_df = pd.read_csv(csv_file, encoding='utf8')
        f, x, y, v = label_df['Frame'].values, label_df['X'].values, label_df['Y'].values, label_df['Visibility'].values
        if split == 'test' and drop:
            rally_key = f'{match_id}_{rally_id}'
            start_f, end_f = start_frame[rally_key], end_frame[rally_key]
            f, x, y, v = f[start_f:end_f] ,x[start_f:end_f], y[start_f:end_f], v[start_f:end_f]
        w, h = Image.open(f'{match_dir}/frame/{rally_id}/0.{IMG_FORMAT}').size
        for i, cx, cy, vis in zip(f, x, y, v):
            image_info.append({'id': sample_count, 'width': w, 'height': h, 'file_name': f'{match_dir}/frame/{rally_id}/{i}.{IMG_FORMAT}'})
            if vis > 0:
                annotations.append({'id': sample_count,
                                    'image_id': sample_count,
                                    'category_id': 1,
                                    'bbox': [int(cx-bbox_size/2), int(cy-bbox_size/2), bbox_size, bbox_size],
                                    'ignore': 0,
                                    'area': bbox_size*bbox_size,
                                    'segmentation': [],
                                    'iscrowd': 0})
            sample_count += 1


    coco_data = {
        'info': {},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'shuttlecock'}],
        'images': image_info,
        'annotations': annotations,
    }
    with open(f'{data_dir}/coco_format_gt.json', 'w') as f:
        json.dump(coco_data, f)
    
################################ Preprocessing Functions ################################
def generate_data_frames(video_file):
    """ Sample frames from the videos in the dataset.

        Args:
            video_file (str): File path of video in dataset
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
        
        Actions:
            Generate frames from the video and save as image files to the corresponding frame directory
    """

    # Check file format
    try:
        assert video_file[-4:] == '.mp4', 'Invalid video file format.'
    except:
        raise ValueError(f'{video_file} is not a video file.')

    # Check if the video has matched csv file
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8')
    assert os.path.exists(video_file) and os.path.exists(csv_file), 'Video file or csv file does not exist.'

    rally_dir = os.path.join(match_dir, 'frame', rally_id)
    if not os.path.exists(rally_dir):
        # Haven't processed yet
        os.makedirs(rally_dir)
    else:
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(rally_dir)) < len(label_df):
            # Some error has occured, remove the directory and process again
            shutil.rmtree(rally_dir)
            os.makedirs(rally_dir)
        else:
            # Already processed.
            return

    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    # Sample frames until video end or exceed the number of labels
    while success and len(frames) != len(label_df):
        success, frame = cap.read()
        if success:
            frames.append(frame)
            cv2.imwrite(os.path.join(rally_dir, f'{len(frames)-1}.{IMG_FORMAT}'), frame)
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)
    median = median[..., ::-1] # BGR to RGB
    np.savez(os.path.join(rally_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def get_match_median(match_dir):
    """ Generate and save the match median frame to the corresponding match directory.

        Args:
            match_dir (str): File path of match directory
                Format: '{data_dir}/{split}/match{match_id}'
            
        Returns:
            None
    """

    medians = []

    # For each rally in the match
    rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
    for rally_dir in rally_dirs:
        file_format_str = os.path.join('{}', 'frame', '{}')
        _, rally_id = parse.parse(file_format_str, rally_dir)

        # Load rally median, if not exist, generate it
        if not os.path.exists(os.path.join(rally_dir, 'median.npz')):
            get_rally_median(os.path.join(match_dir, 'video', f'{rally_id}.mp4'))
        frame = np.load(os.path.join(rally_dir, 'median.npz'))['median']
        medians.append(frame)
    
    # Calculate the median of all rally medians
    median = np.median(np.array(medians), 0)
    np.savez(os.path.join(match_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def get_rally_median(video_file):
    """ Generate and save the rally median frame to the corresponding rally directory.

        Args:
            video_file (str): File path of video file
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
    """
    
    frames = []

    # Get corresponding rally directory
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    save_dir = os.path.join(match_dir, 'frame', rally_id)
    
    # Sample frames from the video
    cap = cv2.VideoCapture(video_file)
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)[..., ::-1] # BGR to RGB
    np.savez(os.path.join(save_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def re_generate_median_files(data_dir):
    for split in ['train', 'val', 'test']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        for match_dir in match_dirs:
            match_name = match_dir.split('/')[-1]
            video_files = list_dirs(os.path.join(match_dir, 'video'))
            for video_file in video_files:
                print(f'Processing {video_file}...')
                get_rally_median(video_file)
            get_match_median(match_dir)
            print(f'Finish processing {match_name}.')
