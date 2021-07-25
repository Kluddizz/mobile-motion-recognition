import os
import glob
import uuid
import shutil
import argparse
from posixpath import join
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def create_labels_file(filename, labels):
  with open(filename, 'w') as f:
    content = '\n'.join(labels)
    f.write(content)


def trim_video_clip(video, duration):
  clips = {}
  num_clips = int(video.duration // duration)

  for i in range(num_clips):
    start = i * duration
    stop = start + duration
    
    id = uuid.uuid4().hex
    clip = video.subclip(start, stop)

    clips[id] = clip

  return clips


if __name__ == '__main__':
  parser = argparse.ArgumentParser(usage='Hello')
  parser.add_argument('--dataset-dir', type=str)
  parser.add_argument('--length', type=int, default=60)
  parser.add_argument('--out-dir', type=str)
  args = parser.parse_args()

  if args.dataset_dir == args.out_dir:
    print('The output directory cannot be the same as the input directory')
    exit(-1)
  elif os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  

  out_labels_file = os.path.join(args.out_dir, 'labels.txt')
  out_annotations_file = os.path.join(args.out_dir, 'annotations.txt')
  out_data_directory = os.path.join(args.out_dir, 'data')

  os.mkdir(args.out_dir)
  os.mkdir(out_data_directory)

  cat_directories = [os.path.join(args.dataset_dir, d) for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]
  labels = [os.path.basename(c) for c in cat_directories]

  create_labels_file(out_labels_file, labels)

  for idx, cd in enumerate(cat_directories):
    vids = glob.glob(os.path.join(cd, '*.mp4'))
    
    for v in vids:
      clip = VideoFileClip(v)
      clipped_videos = trim_video_clip(clip, 2)

      for c in clipped_videos:
        v_filename = os.path.join(out_data_directory, c + '.mp4')
        clipped_videos[c].write_videofile(v_filename)

        with open(out_annotations_file, 'a') as f:
          f.write(f'{v_filename} {idx}\n')

  # ffmpeg_extract_subclip('motions2021/squats/squats1.mp4', 0, 5, 'cut.mp4')