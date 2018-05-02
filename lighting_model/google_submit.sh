#!/bin/bash
BUCKET=gs://render_data
ID=$1
gcloud ml-engine jobs submit training ${ID} \
    --module-name trainer.main \
    --package-path trainer \
    --job-dir ${BUCKET}/out \
    --region europe-west1 \
    --scale-tier BASIC_GPU \
    --runtime-version 1.4 \
    -- \
    --stereo2 \
    --log-dir ${BUCKET}/out/logs \
    --batch-size 16 \
    --learning-rate 1.7e-8 \
    --train-dir ${BUCKET}/scene_data/renders \
    --val-dir ${BUCKET}/scene_data/renders \
    --dotprod False \
    --dotprod-pyramid True \
    --siamese True \
    --multiscale True \
    --log-prefix ${ID}