import pandas as pd
import os

#예시파일입니당. 경로랑 파일은 넣어주세용
df = pd.read_csv(os.path.join('video', 'keyboard.csv'))

#pip install pytube
from pytube import YouTube

#저장 파일 경로만 그때그때 바꿔주세용!
file_path = os.path.join('video', 'keyboard')
for i in range(len(df)):
    print(i, df.iloc[i]['url'])
    yt = YouTube(df.iloc[i]['url'])
    yt_streams = yt.streams
    my_stream = yt_streams.get_by_itag("18")
    my_stream.download(output_path=file_path)
