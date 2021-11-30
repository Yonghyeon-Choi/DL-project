import os

beautyDir = './beauty/'
cookingDir = './cooking/'
footballDir = './football/'
keyboardDir = './keyboard/'
petDir = './pet/'
pokemonDir = './pokemon/'

beautyNames = os.listdir(beautyDir)
cookingNames = os.listdir(cookingDir)
footballNames = os.listdir(footballDir)
keyboardNames = os.listdir(keyboardDir)
petNames = os.listdir(petDir)
pokemonNames = os.listdir(pokemonDir)
beautyNames.sort()
cookingNames.sort()
footballNames.sort()
keyboardNames.sort()
petNames.sort()
pokemonNames.sort()

if '.DS_Store' in beautyNames:
    beautyNames.remove('.DS_Store')
if '.DS_Store' in cookingNames:
    cookingNames.remove('.DS_Store')
if '.DS_Store' in footballNames:
    footballNames.remove('.DS_Store')
if '.DS_Store' in keyboardNames:
    keyboardNames.remove('.DS_Store')
if '.DS_Store' in petNames:
    petNames.remove('.DS_Store')
if '.DS_Store' in pokemonNames:
    pokemonNames.remove('.DS_Store')


def train_test_val_split(content, tot, six):
    return content[0:six], content[six:six+int((tot-six)/2)], content[six+int((tot-six)/2):]


bttrain, btval, bttest = train_test_val_split(beautyNames, 2981, 1789)
cktrain, ckval, cktest = train_test_val_split(cookingNames, 3150, 1890)
fbtrain, fbval, fbtest = train_test_val_split(footballNames, 2447, 1469)
kbtrain, kbval, kbtest = train_test_val_split(keyboardNames, 2416, 1450)
pttrain, ptval, pttest = train_test_val_split(petNames, 2436, 1462)
pktrain, pkval, pktest = train_test_val_split(pokemonNames, 2834, 1700)
train_video = bttrain + cktrain + fbtrain + kbtrain + pttrain + pktrain
print(len(train_video))
val_video = btval + ckval + fbval + kbval + ptval + pkval
print(len(val_video))
test_video = bttest + cktest + fbtest + kbtest + pttest + pktest
print(len(test_video))

import pandas as pd

train = {'video_name': train_video,
         'tag': ['beauty' for _ in range(1789)] +
                ['cooking' for _ in range(1890)] +
                ['football' for _ in range(1469)] +
                ['keyboard' for _ in range(1450)] +
                ['pet' for _ in range(1462)] +
                ['pokemon' for _ in range(1700)]}
val = { 'video_name': val_video,
         'tag': ['beauty' for _ in range(596)] +
                ['cooking' for _ in range(630)] +
                ['football' for _ in range(489)] +
                ['keyboard' for _ in range(483)] +
                ['pet' for _ in range(487)] +
                ['pokemon' for _ in range(567)]}
test = { 'video_name': test_video,
         'tag': ['beauty' for _ in range(596)] +
                ['cooking' for _ in range(630)] +
                ['football' for _ in range(489)] +
                ['keyboard' for _ in range(483)] +
                ['pet' for _ in range(487)] +
                ['pokemon' for _ in range(567)]}
train = pd.DataFrame(train)
val = pd.DataFrame(val)
test = pd.DataFrame(test)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)