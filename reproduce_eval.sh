
#!/bin/bash

set -e


ID_GPU=0
ROOT=./runs

EXPS=(
    ############################
    # Sticky Trap Insects
    ############################
    # DETECTION
    sticky-trap-insects/detection/fasterrcnn/fasterrcnn_256
    sticky-trap-insects/detection/fasterrcnn/fasterrcnn_320
    sticky-trap-insects/detection/fasterrcnn/fasterrcnn_480
    sticky-trap-insects/detection/fasterrcnn/fasterrcnn_640
    sticky-trap-insects/detection/fasterrcnn/fasterrcnn_800
    sticky-trap-insects/detection/fcos/fcos_256
    sticky-trap-insects/detection/fcos/fcos_320
    sticky-trap-insects/detection/fcos/fcos_480
    sticky-trap-insects/detection/fcos/fcos_640
    sticky-trap-insects/detection/fcos/fcos_800
    # DENSITY
    sticky-trap-insects/density/csrnet/csrnet_256
    sticky-trap-insects/density/csrnet/csrnet_320
    sticky-trap-insects/density/csrnet/csrnet_480
    sticky-trap-insects/density/csrnet/csrnet_640
    sticky-trap-insects/density/csrnet/csrnet_800
    sticky-trap-insects/density/fcrn/fcrn_256
    sticky-trap-insects/density/fcrn/fcrn_320
    sticky-trap-insects/density/fcrn/fcrn_480
    sticky-trap-insects/density/fcrn/fcrn_640
    sticky-trap-insects/density/fcrn/fcrn_800
    # SEGMENTATION
    sticky-trap-insects/segmentation/unet/unet_256
    sticky-trap-insects/segmentation/unet/unet_320
    sticky-trap-insects/segmentation/unet/unet_480
    sticky-trap-insects/segmentation/unet/unet_640
    sticky-trap-insects/segmentation/unet/unet_800
)

# Evaluate
for EXP in ${EXPS[@]}; do
    python evaluate.py "$ROOT/experiment=$EXP" --data-root "data/sticky-trap-insects/test" --debug --batch-size 32
done
