import os
from subprocess import call

# CPU 부하량이 크고, 실행시간이 꽤나 깁니다.
Dir = './pokemon/'

Names = os.listdir(Dir)
Names.sort()

for file in Names:
    call('python ffmpeg-split.py -f ' + os.path.join(Dir, file) + ' -s 10 -v h264', shell=True)