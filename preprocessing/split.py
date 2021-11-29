import os
from subprocess import call


Dir = './pokemon/'

Names = os.listdir(Dir)
Names.sort()

for file in Names:
    call('python ffmpeg-split.py -f ' + os.path.join(Dir, file) + ' -s 10 -v h264', shell=True)