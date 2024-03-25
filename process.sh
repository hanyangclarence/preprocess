DATASET_PATH="../dataset/neural_3d_video"
VIDEOS="
coffee_martini
sear_steak
cook_spinach
cut_roasted_beef
flame_salmon_1
flame_steak
"
REF_CAM_ID="00000000"

for video in $VIDEOS; do
  python render_view.py --dataset_path $DATASET_PATH --video $video --reference_camera_id $REF_CAM_ID
done
