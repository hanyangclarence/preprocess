import glob
import cv2

video_dir = 'rendered_results_enerf_zoedepth'
video_list = glob.glob(video_dir + '/feature*')
video_list.sort()

# Output video path
output_video_path = 'output_video.mp4'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Assuming all images are the same size, get the size of the first image
image_size = cv2.imread(video_list[0]).shape[1], cv2.imread(video_list[0]).shape[0]
out = cv2.VideoWriter(output_video_path, fourcc, 1.3, image_size)

for image_path in video_list:
    img = cv2.imread(image_path)
    out.write(img)  # Write out frame to video

# Release everything when job is finished
out.release()