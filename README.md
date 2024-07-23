# BadmintonAnalyticsCV
Badminton Analytics using CV - so far, highlight creation and automatic score updation using TrackNet and YOLO

[TrackNetV3 Model Repository](https://github.com/qaz812345/TrackNetV3)

[YOLOv8 Model Repository](https://github.com/ultralytics/ultralytics)

## Environment setup:

```
pip install -r requirements.txt
```

## Steps to run:

  ### Shuttle Tracking Inference using TrackNetV3:
  1. Execute the following command line statement
     
     ```python3 pre_predict.py --video_file original_short.mp4 (or any other raw footage video)```
  (You can choose to add a --save_dir <dir> argument if you want the predictions to be stored elsewhere and not the default prediction directory)

  ### Using TrackNetV3 predictions for highlights generation and score updation:
  1. Execute the following command line statement
     
     ```python3 testing.py```

After running the above steps, filename_score_clip.mp4 will be created at cwd.


