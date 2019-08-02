# video-human-counting
---

## Purpose
This repository mainly created for counting human(without duplication) in video. 

## Acknowledge
This code is modified from the original code https://github.com/shijieS/SST

## Method
This code mainly set a miss_object pool to store those disappear targets and track failed targets, then if new targets are detected, they would be searched in the miss_object pool to see if these new targets have appeared before. We store the features extracted from the backbone, and then let features pass the affinity estimator to get the similarity array. And this code still have real-time property. So we do not re-train the model, but if you want, you can train by yourself according to the original repo.

## Prepare & Train
Please refer to the original code

## Test
> ```shell
> cd SST-master
> python test_mot17.py
> ```

## Parameters
You can change some parameters in config/config.py.

1.'similar_thresold' means the thresold of new_target and miss_pool object. The lower 'similar_thresold' is tend to have detected more repeat target.

2.'max_object' means the max number of target that allowed in one frame.

## Demo
To be appear.

## Attention
Since this code is based on MOT framework, the counting result is greatly influenced by the performance of detector. You can try different detector in MOT17(DPM/SDP/Faster-RCNN) to test performance of this code.
