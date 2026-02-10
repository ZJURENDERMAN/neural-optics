SCRIPT="optics-simulation-tool/main.py"
SCENE_BASE="../dataset/training_scenes_big"
SIM_CONFIG="simulation_refract_seq_config.json"
OPT_CONFIG="optimization_configs/lens_config_6.json"
TARGET_BASE="../dataset/targets"
OUTPUT_BASE="../results/diff_results_big"

for i in $(seq -w 0 9); do
    SCENE_DIR="$SCENE_BASE/scenes_$i/refraction/bspline"
    SEED=${i#0}  # 去掉前导零
    
    for target in letters digits; do
        python $SCRIPT $SCENE_DIR \
            --sim $SIM_CONFIG \
            --opt $OPT_CONFIG \
            --target-dir $TARGET_BASE/$target \
            --output-dir $OUTPUT_BASE/${target}_results$i/ \
            --match-targets --match-seed $SEED
    done
done