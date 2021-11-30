import pandas as pd
import os

df = pd.read_csv(os.path.join('csv', 'keyboard.csv'))

# pip install pytube
from pytube import YouTube

file_path = os.path.join('video', 'keyboard')
for i in range(len(df)):
    print(i, df.iloc[i]['url'])
    yt = YouTube(df.iloc[i]['url'])
    yt_streams = yt.streams
    my_stream = yt_streams.get_by_itag("18") # 18 means 320p
    my_stream.download(output_path=file_path)
