SCRIPT="../optics-simulation-tool/main.py"
SCENE_BASE="$HOME/neural-optics-dataset/scenes"
SIM_CONFIG="simulation-config.json"
OPT_CONFIG="optimization-config.json"
TARGET_BASE="$HOME/neural-optics-dataset/targets"
OUTPUT_BASE="$HOME/neural-optics-result/optimization_results"

for i in $(seq -w 0 9); do
    SCENE_DIR="$SCENE_BASE/scenes_$i/refraction/bspline"
    
    for target in letters digits; do
        python $SCRIPT $SCENE_DIR \
            --sim $SIM_CONFIG \
            --opt $OPT_CONFIG \
            --target-dir $TARGET_BASE/$target \
            --output-dir $OUTPUT_BASE/${target}_results$i/ \
            --match-targets --match-seed $i
    done
done