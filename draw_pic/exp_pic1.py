import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#下面两句代码防止中文显示成方块
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


# plt.rcParams['figure.figsize'] = (10.0, 4.0) # 设置figure_size尺寸
plt.rcParams['figure.dpi'] = 300 #图片像素

x_bits = [12, 24, 32, 48]
plt.rcParams['figure.figsize'] = (12.0, 4.0)
def joint_early():

    with PdfPages('joint_early.pdf') as pdf:
        #cifar10-----------------------------------
        plt.subplot(1,2, 1)
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
        labels_early=['student-1-early','student-2-early',
                      'student-3-early','student-4-early']
        labels_ours=['student-1-ours','student-2-ours',
                     'student-3-ours','student-4-ours']
        for i in range(4):
            color=colors[i]
            plt.plot(x_bits,y_early_students[i],color=color,
                     marker=markers[i],linestyle='--',label=labels_early[i])
            plt.plot(x_bits,y_ours_students[i],color=color,
                     marker=markers[i],linestyle='-',label=labels_ours[i])
        plt.xticks(x_bits)  #横轴只有这四个刻度
        # plt.ylim(0.65, 0.93)       #y坐标范围
        plt.title("CIFAR-10")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签

        #sun-----------------------------------
        plt.subplot(1,2, 2)
        # 数据准备
        y_early_students = [[0.75348, 0.8166, 0.82471, 0.82749],
                            [0.78058, 0.83077, 0.83574, 0.84841],
                            [0.8249, 0.86114, 0.8627, 0.87235],
                            [0.83924, 0.86368, 0.8674, 0.87599]
                            ]
        y_ours_students = [[0.76123, 0.81592, 0.82724, 0.84166],
                           [0.78617, 0.83835, 0.84371, 0.85208],
                           [0.82594, 0.86230, 0.86595, 0.87225],
                           [0.83931, 0.86902, 0.87141, 0.87521]]

        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green']
        markers = ['o', 'v', '*', 's']
        labels_early = ['student-1-early', 'student-2-early',
                        'student-3-early', 'student-4-early']
        labels_ours = ['student-1-ours', 'student-2-ours',
                       'student-3-ours', 'student-4-ours']
        for i in range(4):
            color = colors[i]
            plt.plot(x_bits, y_early_students[i], color=color,
                     marker=markers[i], linestyle='--', label=labels_early[i])
            plt.plot(x_bits, y_ours_students[i], color=color,
                     marker=markers[i], linestyle='-', label=labels_ours[i])

        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # plt.ylim(0.7, 0.9)       #y坐标范围
        plt.title("SUN")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
        plt.legend()
        # plt.show()
        pdf.savefig( bbox_inches='tight')  # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉

def joint_slim():

    with PdfPages('joint_slim.pdf') as pdf:
        #cifar10-----------------------------------
        plt.subplot(1,2, 1)
        y_slim_widths = [[0.83371, 0.85823, 0.87375, 0.8732],
                         [0.87009, 0.89147, 0.89757, 0.90113],
                         [0.87441, 0.89441, 0.90316, 0.90495],
                         [0.87696, 0.89471, 0.90049, 0.90572]
                         ]
        y_ours_widths = [[0.85901, 0.87269, 0.8836, 0.88728],
                         [0.87894, 0.89516, 0.90348, 0.90875],
                         [0.87636, 0.8966, 0.90935, 0.91106],
                         [0.87955, 0.89755, 0.91043, 0.91302]]

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
        plt.title("CIFAR-10")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签

        #sun-----------------------------------
        plt.subplot(1, 2, 2)
        y_slim_widths = [[0.79963, 0.83232, 0.83891, 0.84012],
                         [0.83025, 0.85247, 0.86035, 0.8674],
                         [0.83533, 0.8553, 0.86254, 0.86971],
                         [0.83407, 0.85779, 0.86261, 0.87181]
                         ]
        y_ours_widths = [[0.81997, 0.84620, 0.85041, 0.85036],
                         [0.83417, 0.86393, 0.86908, 0.87413],
                         [0.84109, 0.86835, 0.87002, 0.87449],
                         [0.83931, 0.86902, 0.87141, 0.87521]]

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
        pdf.savefig( bbox_inches='tight') # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉


#PNT============================
def PNT_early():
    with PdfPages('PNT_early.pdf') as pdf:
        # cifar10==========================
        plt.subplot(1, 2, 1)
        y_baseline1_students = [[0.70507,0.74372,0.75134,0.75316],
                            [0.72906,0.75637,0.76007,0.77722],
                            [0.76451,0.78874,0.78934,0.79695],
                            [0.77135,0.79342,0.79956,0.80243]
                            ]
        y_baseline2_students = [[0.7426,0.77574,0.77362,0.79066],
                                [0.80025,0.82003,0.81916,0.83368],
                                [0.86393,0.88676,0.89845,0.89881],
                                [0.87718,0.89152,0.90806,0.90779]
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
        plt.title("CIFAR-10")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签


        #sun==============
        plt.subplot(1, 2, 2)
        y_baseline1_students = [[0.72762,0.78089,0.79606,0.80523],
                                [0.74106,0.79834,0.80569,0.81506],
                                [0.77537,0.81288,0.82147,0.82639],
                                [0.78615,0.81925,0.82477,0.8297]
                                ]
        y_baseline2_students = [[0.74154,0.79641,0.81081,0.83131],
                                [0.77466,0.82422,0.82667,0.84823],
                                [0.823,0.85487,0.86119,0.87041],
                                [0.84009,0.86671,0.86941,0.87294]
                                ]

        y_PNT_students = [[0.76123,0.81592,0.82724,0.84166],
                         [0.78617,0.83835,0.84371,0.85208],
                         [0.82594,0.86230,0.86595,0.87225],
                         [0.83931,0.86902,0.87141,0.87521]]

        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green']
        markers = ['o', 'v', '*', 's']
        labels_baseline1 = ['student-1-scratch', 'student-2-scratch', 'student-3-scratch', 'student-4-scratch']
        labels_baseline2 = ['student-1-trunk', 'student-2-trunk', 'student-3-trunk', 'student-4-trunk']
        labels_PNT = ['student-1-PNT', 'student-2-PNT', 'student-3-PNT', 'student-4-PNT']
        for i in range(4):
            color = colors[i]
            plt.plot(x_bits, y_baseline1_students[i], color=color, marker=markers[i], linestyle=':',
                     label=labels_baseline1[i])
            plt.plot(x_bits, y_baseline2_students[i], color=color, marker=markers[i], linestyle='--',
                     label=labels_baseline2[i])
            plt.plot(x_bits, y_PNT_students[i], color=color, marker=markers[i], linestyle='-', label=labels_PNT[i])

        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # plt.ylim(0.65, 0.93)       #y坐标范围
        plt.title("SUN")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.show()
        pdf.savefig( bbox_inches='tight')

def PNT_slim():
    with PdfPages('PNT_slim.pdf') as pdf:
        # cifar10==========================
        plt.subplot(1, 2, 1)
        y_baseline1_widths = [[0.73202,0.75943,0.76133,0.76706],
                            [0.76767,0.77712,0.78871,0.77915],
                            [0.77645,0.7813,0.81148,0.79121],
                            [0.77135,0.79342,0.79956,0.80243]
                            ]
        y_baseline2_widths = [[0.85834,0.87265,0.89302,0.88718],
                                [0.87809,0.89142,0.9027,0.90023],
                                [0.8766,0.89071,0.90558,0.90239],
                                [0.87718,0.89152,0.90806,0.90779]
                                ]

        y_PNT_widths = [[0.85901, 0.87269, 0.8836, 0.88728],
                         [0.87894, 0.89516, 0.90348, 0.90875],
                         [0.87636, 0.8966, 0.90935, 0.91106],
                         [0.87955, 0.89755, 0.91043, 0.91302]]

        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green']
        markers = ['o', 'v', '*', 's']
        labels_baseline1 = ['width0.25x-scratch', 'width0.5x-scratch', 'width0.75x-scratch', 'width1.0x-scratch']
        labels_baseline2 = ['width0.25x-trunk', 'width0.5x-trunk', 'width0.75x-trunk', 'width1.0x-trunk']
        labels_PNT = ['width0.25x-PNT', 'width0.5x-PNT', 'width0.75x-PNT', 'width1.0x-PNT']
        for i in range(4):
            color = colors[i]
            plt.plot(x_bits, y_baseline1_widths[i], color=color, marker=markers[i], linestyle=':', label=labels_baseline1[i])
            plt.plot(x_bits, y_baseline2_widths[i], color=color, marker=markers[i], linestyle='--', label=labels_baseline2[i])
            plt.plot(x_bits, y_PNT_widths[i], color=color, marker=markers[i], linestyle='-', label=labels_PNT[i])

        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # plt.ylim(0.65, 0.93)       #y坐标范围
        plt.title("CIFAR-10")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签


        #sun==============
        plt.subplot(1, 2, 2)
        y_baseline1_widths = [[0.73827,0.77496,0.78279,0.79029],
                              [0.77087,0.80415,0.8054,0.80703],
                              [0.78025,0.81502,0.81836,0.82217],
                              [0.78615,0.81925,0.82477,0.8297]
                              ]
        y_baseline2_widths = [[0.81453,0.84397,0.8489,0.85087],
                              [0.83554,0.86239,0.8671,0.86977],
                              [0.84063,0.86526,0.86748,0.8748],
                              [0.84009,0.86671,0.86941,0.87294]
                              ]

        y_PNT_widths =  [[0.81997, 0.84620, 0.85041, 0.85036],
                         [0.83417, 0.86393, 0.86908, 0.87413],
                         [0.84109, 0.86835, 0.87002, 0.87449],
                         [0.83931, 0.86902, 0.87141, 0.87521]]

        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green']
        markers = ['o', 'v', '*', 's']
        labels_baseline1 = ['width0.25x-scratch', 'width0.5x-scratch', 'width0.75x-scratch', 'width1.0x-scratch']
        labels_baseline2 = ['width0.25x-trunk', 'width0.5x-trunk', 'width0.75x-trunk', 'width1.0x-trunk']
        labels_PNT = ['width0.25x-PNT', 'width0.5x-PNT', 'width0.75x-PNT', 'width1.0x-PNT']
        for i in range(4):
            color = colors[i]
            plt.plot(x_bits, y_baseline1_widths[i], color=color, marker=markers[i], linestyle=':',
                     label=labels_baseline1[i])
            plt.plot(x_bits, y_baseline2_widths[i], color=color, marker=markers[i], linestyle='--',
                     label=labels_baseline2[i])
            plt.plot(x_bits, y_PNT_widths[i], color=color, marker=markers[i], linestyle='-', label=labels_PNT[i])

        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # plt.ylim(0.65, 0.93)       #y坐标范围
        plt.title("SUN")
        plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.show()
        pdf.savefig( bbox_inches='tight')

# joint_early()
# joint_slim()
# PNT_early()
PNT_slim()