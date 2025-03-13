# BadmintonAnalyticsCV

Badminton Analytics using CV - so far, highlight creation and automatic score updation using TrackNet and YOLO.

TrackNetV3 Model Repository

YOLOv8 Model Repository

## Environment setup

To set up the environment, run the following command:

```
pip install -r requirements.txt
```

## Steps to run

### Shuttle Tracking Inference using TrackNetV3

1. Execute the following command line statement:
   ```bash
   python3 pre_predict.py --video_file original_short.mp4
   ```
   (You can choose to add a `--save_dir <dir>` argument if you want the predictions to be stored elsewhere and not the default `prediction` directory)

### Using TrackNetV3 predictions for highlights generation and score updation

1. Execute the following command line statement:
   ```bash
   python3 testing.py
   ```

After running the above steps, `filename_score_clip.mp4` will be created at the current working directory.

## Repository structure

- `.gitattributes`: Git LFS configuration for `.pt` files.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `ckpts/`: Directory containing model checkpoint files.
  - `ckpts/best.pt`
  - `ckpts/InpaintNet_best.pt`
  - `ckpts/TrackNet_best.pt`
- `dataset.py`: Contains the `Shuttlecock_Trajectory_Dataset` and `Video_IterableDataset` classes for handling dataset operations.
- `model.py`: Defines the `TrackNet` and `InpaintNet` models.
- `pre_predict.py`: Script for preprocessing and predicting shuttle tracking using TrackNetV3.
- `predict.py`: Contains functions for predicting shuttle locations and handling video processing.
- `prediction/`: Directory for storing prediction results.
  - `prediction/original_short_ball.csv` SKIPPED
  - `prediction/placeholder_file`
- `README.md`: This file.
- `requirements.txt`: Lists the required Python packages.
- `test.py`: Contains functions for evaluating the models.
- `testing.py`: Script for generating highlights and updating scores using TrackNetV3 predictions.
- `utils/`: Directory containing utility functions.
  - `utils/__init__.py`
  - `utils/func_clips_start_end.py`
  - `utils/general.py` SKIPPED
  - `utils/left_right.py`
  - `utils/metric.py`
  - `utils/ullasmodel.py`

## Additional information

### TrackNetV3 Model

TrackNetV3 is a deep learning model for tracking shuttlecock trajectories in badminton videos. It consists of two main components:
- `TrackNet`: A convolutional neural network for predicting heatmaps of shuttlecock positions.
- `InpaintNet`: A neural network for inpainting missing shuttlecock positions in the predicted trajectories.

### YOLOv8 Model

YOLOv8 is a state-of-the-art object detection model used for detecting players and other objects in badminton videos.

### Dataset

The dataset used for training and evaluating the models consists of badminton match videos with annotated shuttlecock positions. The `Shuttlecock_Trajectory_Dataset` class in `dataset.py` handles the loading and preprocessing of the dataset.
