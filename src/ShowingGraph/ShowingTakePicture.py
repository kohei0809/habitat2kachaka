import pandas as pd
import matplotlib.pyplot as plt
import pathlib

date = "23-08-27 21-24-26"
#date = "23-09-03 02-46-00"
date = "23-09-29 09-11-11"
mode = "train"
#mode = "val"

df = pd.read_csv("log/" + date + "/" + mode + "/metrics.csv", names=['time', 'picture', 'ci', 'exp_area', 'path_length'], header=None)
plt.plot(df['time'], df['picture'], color="blue", label="Number of times taken pictures")

#ラベルの追加
plt.xlabel('Training Steps')
plt.ylabel('Number of Times Taken Pictures')

#表示範囲の指定
#plt.xlim(0, 50000000)
#plt.ylim(0, 1.0)

#凡例の追加
plt.legend()

#指数表記から普通の表記に変換
plt.ticklabel_format(style='plain',axis='x')
plt.ticklabel_format(style='plain',axis='y')

#フォルダがない場合は、作成
p_dir = pathlib.Path("./result/" + mode + "/taken_picture_graph")
if not p_dir.exists():
    p_dir.mkdir(parents=True)

#グラフの保存
plt.savefig('./result/' + mode + '/taken_picture_graph/' + date + '.png')

#グラフの表示
plt.show()

print("Showing Taken Picture graph is completed.")