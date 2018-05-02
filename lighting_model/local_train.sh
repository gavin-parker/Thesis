#!/bin/bash
BUCKET=gs://render_data
gcloud ml-engine local train \
    --module-name trainer.main \
    --package-path trainer \
    --job-dir . \
    --runtime-version 1.4 \
    --\
    --stereo2 \
    --log-dir ${BUCKET}/logs/ \
    --batch-size 1 \
    --learning-rate 1.7e-8 \
    --train-dir ${BUCKET}/scene_data/renders \
    --val-dir ${BUCKET}/scene_data/val
