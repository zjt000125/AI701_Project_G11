source ~/local/miniconda3/etc/profile.d/conda.sh
conda activate xmem2
mkdir ./key_frames/
ffmpeg -i ./videos/three_cats.mp4 \
       -filter:v "select='gt(scene,0.05)',showinfo" \
       -vsync 0 ./key_frames/frame_%06d.jpg

python ./pipline/show_key_frames.py

mkdir ../XMem2/example_videos/demo/
mv ./key_frames/ ../XMem2/example_videos/demo/JPEGImages/

cd ../XMem2/
python interactive_demo.py --images example_videos/demo/JPEGImages

python ./util/mask_process.py
python ./util/Images2Segments.py

cd ../Ai701_Project_G11/

python ./pipline/select_reference_object.py

# source ~/local/miniconda3/etc/profile.d/conda.sh
# conda activate fast

# bash generate.sh

rm -rf ../XMem2/example_videos/demo/
rm -rf ../XMem2/workspace/demo/