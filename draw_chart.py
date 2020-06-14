import matplotlib.pyplot as plt
import numpy as np
from transform_functions import rot2euler


def create_label(array):
    labels = {}
    for i in array:
        num = labels.get(i, 0)
        num += 1
        labels[i] = num
    return labels

datafile = np.loadtxt('./dataset_files/parsed_set_normal.txt', dtype=str)
test_data = datafile[-3000:]
translation = np.zeros([1], dtype=np.float)
rotation = np.zeros([1], dtype=np.float)
for t in test_data:
    T = t[4:]
    T = T.astype(np.float)
    T = np.reshape(T, [4, 4])
    tr = T[0:3, 3]
    ro = rot2euler(T[0:3, 0:3]) / np.pi * 180

    translation = np.hstack([translation, tr])
    rotation = np.hstack([rotation, ro])
    # print(tr)
    # print(ro / np.pi * 180)
translation = np.round(np.abs(translation[1:]), decimals=2)
rotation = np.round(np.abs(rotation[1:]), decimals=2)
print(np.min(translation), np.max(translation))
print(np.min(rotation), np.max(rotation))

translation_dict = create_label(rotation)
x = list(translation_dict.keys())
x = sorted(x)
y = []
for i in x:
    y.append(translation_dict[i])
print(x)
print(y)

# plt.bar(x, y2, label="label2",color='orange')
# plt.bar(x, y3, label="label3", color='lightgreen')
#
plt.xticks(np.arange(len(x)), x, rotation=0, fontsize=8)  # 数量多可以采用270度，数量少可以采用340度，得到更好的视图
plt.hist(rotation, bins=x, color='red', width=0.01, edgecolor="black")
plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('Numbers')
plt.xlabel('Errors/°')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.rcParams['figure.figsize'] = (100.0, 80.0)  # 尺寸
plt.title("Rotation Error")
plt.show()


