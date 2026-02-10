SCRIPT="../optics-scene-tool/scene_data_generator.py"
NUM=10000
OUTPUT_DIR=~/neural-optics-dataset/scenes

for i in $(seq -w 0 9); do
    python $SCRIPT --num $NUM --output $OUTPUT_DIR/scenes_$i --seed $i --visualize
done