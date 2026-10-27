cd /e/grasp_video
E:/envs/manim/Scripts/manim.exe --resolution 1920,1080 --frame_rate 30 --disable_caching grasp_video.py -a > render_voiced.log 2>&1
D=media/videos/grasp_video/1080p30
: > list_v.txt
for s in S01Title S02Molecule S03Tokens S04Vector S05Attention S06Twins S07Superposition S08SAE S09Test S10Verdict; do echo "file '$D/$s.mp4'" >> list_v.txt; done
ffmpeg -loglevel error -y -f concat -safe 0 -i list_v.txt -c:v copy -c:a aac -b:a 160k -ar 48000 GRASP_phenol_feature_narrated.mp4 && echo FINAL_OK >> render_voiced.log
