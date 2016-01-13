#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=1
NET=VGG16
NET_lc=${NET,,}
ITERS=600000
DATASET_TRAIN=coco_2014_train
DATASET_TEST=coco_2014_val
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
NET_FINAL=output/faster_rcnn_end2end/coco_train2014/coco_vgg16_faster_rcnn_iter_565000.caffemodel




time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/faster_rcnn_end2end/coco_test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
