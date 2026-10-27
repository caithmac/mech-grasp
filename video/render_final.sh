cd /e/grasp_video
E:/envs/manim/Scripts/manim.exe --resolution 1920,1080 --frame_rate 30 --disable_caching grasp_video.py -a > render_final.log 2>&1
D=media/videos/grasp_video/1080p30
: > list.txt
for s in S01Title S02Molecule S03Tokens S04Vector S05Attention S06Twins S07Superposition S08SAE S09Test S10Verdict; do echo "file '$D/$s.mp4'" >> list.txt; done
ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c copy GRASP_phenol_feature.mp4 && echo FINAL_OK >> render_final.log
