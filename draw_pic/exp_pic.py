import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#下面两句代码防止中文显示成方块
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


# plt.rcParams['figure.figsize'] = (10.0, 4.0) # 设置figure_size尺寸
plt.rcParams['figure.dpi'] = 300 #图片像素

x_bits = [12, 24, 32, 48]
'''
验证joint_training
'''
def joint_early_cifar():
    #cifar10
    #数据准备
    y_early_students=[[0.741551,0.76498,0.77793,0.77558],
                      [0.79482,0.818521,0.82154,0.827666],
                      [0.86727,0.89018,0.90012,0.900969],
                      [0.87354,0.8927,0.90919,0.905798]
                      ]
    y_ours_students=[[0.76229,0.79393,0.78994,0.80432],
                     [0.80788,0.82604,0.84254,0.83907],
                     [0.87004,0.89094,0.90188,0.9048],
                     [0.87955,0.89755,0.91043,0.91302]]

    #开始绘制
    colors=['red','orange','blue','green']
    markers=['o','v','*','s']
    labels_early=['student-1-early','student-2-early','student-3-early','student-4-early']
    labels_ours=['student-1-ours','student-2-ours','student-3-ours','student-4-ours']
    for i in range(4):

        color=colors[i]
        plt.plot(x_bits,y_early_students[i],color=color,marker=markers[i],linestyle='--',label=labels_early[i])
        plt.plot(x_bits,y_ours_students[i],color=color,marker=markers[i],linestyle='-',label=labels_ours[i])

    plt.xticks(x_bits)  #横轴只有这四个刻度
    # plt.ylim(0.65, 0.93)       #y坐标范围
    plt.title("CIFAR10")
    plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
    # plt.legend()
    plt.show()

def joint_early_sun():
    #sun42

    with PdfPages('joint_early_sun.pdf') as pdf:
        #数据准备
        y_early_students=[[0.75348,0.8166,0.82471,0.82749],
                          [0.78058,0.83077,0.83574,0.84841],
                          [0.8249,0.86114,0.8627,0.87235],
                          [0.83924,0.86368,0.8674,0.87599]
                          ]
        y_ours_students=[[0.76123,0.81592,0.82724,0.84166],
                         [0.78617,0.83835,0.84371,0.85208],
                         [0.82594,0.86230,0.86595,0.87225],
                         [0.83931,0.86902,0.87141,0.87521]]

        #开始绘制
        colors=['red','orange','blue','green']
        markers=['o','v','*','s']
        labels_early=['student-1-early','student-2-early','student-3-early','student-4-early']
        labels_ours=['student-1-ours','student-2-ours','student-3-ours','student-4-ours']
        for i in range(4):
            color=colors[i]
            plt.plot(x_bits,y_early_students[i],color=color,marker=markers[i],linestyle='--',label=labels_early[i])
            plt.plot(x_bits,y_ours_students[i],color=color,marker=markers[i],linestyle='-',label=labels_ours[i])

        plt.xticks(x_bits)  #横轴只有这四个刻度
        # plt.ylim(0.7, 0.9)       #y坐标范围
        plt.title("SUN")
        plt.xlabel("Number of bits")#作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")#作用为纵坐标轴添加标签
        plt.legend()
        # plt.show()
        pdf.savefig() #如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉

def joint_slim_cifar():
    # cifar10
    # 数据准备
    y_slim_widths = [[0.83371,0.85823,0.87375,0.8732],
                        [0.87009,0.89147,0.89757,0.90113],
                        [0.87441,0.89441,0.90316,0.90495],
                        [0.87696,0.89471,0.90049,0.90572]
                        ]
    y_ours_widths = [[0.85901,0.87269,0.8836,0.88728],
                       [0.87894,0.89516,0.90348,0.90875],
                       [0.87636,0.8966,0.90935,0.91106],
                       [0.87955,0.89755,0.91043,0.91302]]

    # 开始绘制
    colors = ['red', 'orange', 'blue', 'green']
    markers = ['o', 'v', '*', 's']
    labels_early = ['width0.25x-slim', 'width0.5x-slim', 'width0.75x-slim', 'width1.0x-slim']
    labels_ours = ['width0.25x-ours', 'width0.5x-ours', 'width0.75x-ours', 'width1.0x-ours']
    for i in range(4):
        color = colors[i]
        plt.plot(x_bits, y_slim_widths[i], color=color, marker=markers[i], linestyle='--', label=labels_early[i])
        plt.plot(x_bits, y_ours_widths[i], color=color, marker=markers[i], linestyle='-', label=labels_ours[i])

    plt.xticks(x_bits)  # 横轴只有这四个刻度
    # plt.ylim(0.65, 0.93)       #y坐标范围
    plt.title("CIFAR10")
    plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
    # plt.legend()
    plt.show()

def joint_slim_sun():
    # cifar10
    # 数据准备
    y_slim_widths = [[0.79963,0.83232,0.83891,0.84012],
                        [0.83025,0.85247,0.86035,0.8674],
                        [0.83533,0.8553,0.86254,0.86971],
                        [0.83407,0.85779,0.86261,0.87181]
                        ]
    y_ours_widths = [[0.81997,0.84620,0.85041,0.85036],
                       [0.83417,0.86393,0.86908,0.87413],
                       [0.84109,0.86835,0.87002,0.87449],
                       [0.83931,0.86902,0.87141,0.87521]]

    # 开始绘制
    colors = ['red', 'orange', 'blue', 'green']
    markers = ['o', 'v', '*', 's']
    labels_early = ['width0.25x-slim', 'width0.5x-slim', 'width0.75x-slim', 'width1.0x-slim']
    labels_ours = ['width0.25x-ours', 'width0.5x-ours', 'width0.75x-ours', 'width1.0x-ours']
    for i in range(4):
        color = colors[i]
        plt.plot(x_bits, y_slim_widths[i], color=color, marker=markers[i], linestyle='--', label=labels_early[i])
        plt.plot(x_bits, y_ours_widths[i], color=color, marker=markers[i], linestyle='-', label=labels_ours[i])

    plt.xticks(x_bits)  # 横轴只有这四个刻度
    # plt.ylim(0.65, 0.93)       #y坐标范围
    plt.title("SUN")
    plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
    plt.legend()
    plt.show()



def PNT_early_cifar():
    # cifar10
    # 数据准备
    y_baseline1_students = [[0.741551, 0.76498, 0.77793, 0.77558],
                        [0.79482, 0.818521, 0.82154, 0.827666],
                        [0.86727, 0.89018, 0.90012, 0.900969],
                        [0.87354, 0.8927, 0.90919, 0.905798]
                        ]
    y_baseline2_students = [[0.741551, 0.76498, 0.77793, 0.77558],
                            [0.79482, 0.818521, 0.82154, 0.827666],
                            [0.86727, 0.89018, 0.90012, 0.900969],
                            [0.87354, 0.8927, 0.90919, 0.905798]
                            ]

    y_PNT_students = [[0.76229, 0.79393, 0.78994, 0.80432],
                       [0.80788, 0.82604, 0.84254, 0.83907],
                       [0.87004, 0.89094, 0.90188, 0.9048],
                       [0.87955, 0.89755, 0.91043, 0.91302]]

    # 开始绘制
    colors = ['red', 'orange', 'blue', 'green']
    markers = ['o', 'v', '*', 's']
    labels_baseline1 = ['student-1-scratch', 'student-2-scratch', 'student-3-scratch', 'student-4-scratch']
    labels_baseline2 = ['student-1-trunk', 'student-2-trunk', 'student-3-trunk', 'student-4-trunk']
    labels_PNT = ['student-1-PNT', 'student-2-PNT', 'student-3-PNT', 'student-4-PNT']
    for i in range(4):
        color = colors[i]
        plt.plot(x_bits, y_baseline1_students[i], color=color, marker=markers[i], linestyle=':', label=labels_baseline1[i])
        plt.plot(x_bits, y_baseline2_students[i], color=color, marker=markers[i], linestyle='--', label=labels_baseline2[i])
        plt.plot(x_bits, y_PNT_students[i], color=color, marker=markers[i], linestyle='-', label=labels_PNT[i])

    plt.xticks(x_bits)  # 横轴只有这四个刻度
    # plt.ylim(0.65, 0.93)       #y坐标范围
    plt.title("CIFAR10")
    plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
    # plt.legend()
    plt.show()





# joint_early_sun()
