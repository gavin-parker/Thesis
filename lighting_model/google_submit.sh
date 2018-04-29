#!/bin/bash
BUCKET=$1

gcloud ml-engine jobs submit training stereo_test2 \
    --module-name trainer.main \
    --package-path trainer/ \
    --job-dir ${BUCKET}/out \
    --region europe-west1 \
    -- \
    --stereo2 \
    --log-dir ${BUCKET}/logs/ \
    --batch-size 32 \
    --learning-rate 1.7e-8 \
    --train-dir ${BUCKET}/scene_data/val \
    --val-dir ${BUCKET}/scene_data/val