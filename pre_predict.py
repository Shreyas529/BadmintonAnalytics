from predict import *
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--save_dir', default = 'prediction', type = str)
    args = parser.parse_args()
    save_dir = args.save_dir
    video_name = args.video_file.split('/')[-1][:-4]
    out_csv_file = os.path.join(save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(save_dir, f'{video_name}.mp4')

    cap = cv2.VideoCapture(args.video_file)
    
    # converting video to 30 fps if it isn't already
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    except:
        print("Error in calculating FPS")

    cap.release()

    if (fps != 30): 
        change_fps(args.video_file)

    
    # predicting in batches to not use too much memory

    vfc = VideoFileClip(args.video_file)
    clip_duration = 10
    total_length = vfc.duration
    num_clips = int(total_length // clip_duration)

    remainder = total_length - num_clips*clip_duration
    # print(remainder)

    if remainder >= 1:
        num_clips += 1

    # print(num_clips)
    pred_dict_joined = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    for i in range(num_clips):
        if (i != num_clips-1):
            clip = vfc.subclip(i * clip_duration, (i+1) * clip_duration)
            clip.write_videofile("temp_clip.mp4")
            cap = cv2.VideoCapture("temp_clip.mp4")

            frame_list, fps, (w,h) = generate_frames_from_cap(cap)
            pred_dict = pred_main(frame_list=frame_list, fps=fps, w=w, h=h)
        
        else:
            clip = vfc.subclip(i * clip_duration, total_length)
            clip.write_videofile("temp_clip.mp4")
            cap = cv2.VideoCapture("temp_clip.mp4")
            frame_list, fps, (w,h) = generate_frames_from_cap(cap)
            pred_dict = pred_main(frame_list=frame_list, fps=fps, w=w, h=h)

        
        for k in range(len(pred_dict['Frame'])):
            pred_dict['Frame'][k] += i*30*clip_duration

        pred_dict_joined['Frame'].extend(pred_dict['Frame'])
        pred_dict_joined['Visibility'].extend(pred_dict['Visibility'])
        pred_dict_joined['X'].extend(pred_dict['X'])
        pred_dict_joined['Y'].extend(pred_dict['Y'])

        # print(pred_dict_joined)
        
    pred_df = pandas.DataFrame({'Frame': pred_dict_joined['Frame'],
                                'Visibility': pred_dict_joined['Visibility'],
                                'X': pred_dict_joined['X'],
                                'Y': pred_dict_joined['Y']})
    
    pred_df.to_csv(out_csv_file, index=False)

    with open(f'predicted.bin','wb') as file:
        pickle.dump(pred_dict_joined, file)
        pickle.dump(out_video_file, file)
        pickle.dump(args.video_file, file)
