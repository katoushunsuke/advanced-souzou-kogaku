import os
import glob
import cv2
import pandas as pd
import mediapipe as mp

df = []
for foldername in os.listdir('./hand/'):
    imgs_path = './hand/' + foldername
    imgs = sorted(glob.glob(imgs_path + '/' + '*.jpg'))
    for name in imgs:
        df.append((str(name), str(foldername)))
df = pd.DataFrame(df, columns=['img', 'label'])
df.head()

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

df1 = []
for idx, file in enumerate(df['img']):
    print('No', idx)
    image = cv2.flip(cv2.imread(file), 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        print('No Hand')
        print('-----------------------')
        continue
    mark_list = []

    for i in range(21):

        x = results.multi_hand_landmarks[0].landmark[i].x
        y = results.multi_hand_landmarks[0].landmark[i].y
        z = results.multi_hand_landmarks[0].landmark[i].z

        mark_list.append(x)
        mark_list.append(y)
        mark_list.append(z)

    mark_list.append(df['label'][idx])
    df1.append(mark_list)
    print('complete')
    print('-------------------------')

df1 = pd.DataFrame(df1)
df1.shape

df2 = []
for idx, file in enumerate(df['img']):
    print('No', idx)
    image = cv2.imread(file, 1)  # flipしない→逆の手としてデータとれるはず
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        print('No Hand')
        print('-------------------------')
        continue
    mark_list = []
    for i in range(21):
        x = results.multi_hand_landmarks[0].landmark[i].x
        y = results.multi_hand_landmarks[0].landmark[i].y
        z = results.multi_hand_landmarks[0].landmark[i].z

        mark_list.append(x)
        mark_list.append(y)
        mark_list.append(z)
    mark_list.append(df['label'][idx])
    df2.append(mark_list)
    print('-------------------------')

df2 = pd.DataFrame(df2)
df2.shape
# ---> (598, 64)

df3 = pd.concat([df1, df2])
df3.shape
# ---> (1195, 64)
df3.head()

df3.to_csv('landmarkdata2.csv', index=False)
