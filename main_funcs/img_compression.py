from PIL import Image as Img
import os
import pandas as pd
import numpy as np
#поменять строки только здесь и в крайней строке
folder_path = 'img_for_train/j_letter/j_letter_1'
cnt = 0
col_names = np.array(['label' if x == 0 else ('pixel' + str(x)) for x in range(785)])
df = pd.DataFrame(np.zeros((1,785), dtype='int'), columns=col_names)

for file in os.listdir(folder_path):
    img = Img.open(folder_path + '/' + file)
    img = img.resize((28,28))
    img_array = np.array(img)
    # Преобразуем массив в одномерный массив (784 элемента)
    img_flattened = img_array.flatten()
    img_flattened = np.insert(img_flattened, 0, 10)
    df.loc[cnt] = img_flattened
    cnt += 1

print(df)
hello_csv = df.to_csv('j_2.csv')