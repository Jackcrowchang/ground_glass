import tkinter as tk
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import tkinter.filedialog
from numpy import fft
import math
import matplotlib.pyplot as graph


window = tk.Tk()
window.title('毛玻璃清晰化处理软件')
window.geometry('815x790')
address0 = 'code_image//timg.jpg'
address1 = 'code_image//timg.jpg'
folder = 1
photo0 = None
photo1 = None
contrast_data0 = 5
contrast_data1 = 1.3
denosing_data = 10
expansion_data = 1
rust_data = 1
exposure = -5
logic = 1
winner_data = 0.001

global img, img1
global point1, point2, point1_dis, point2_dis


def resizeImage(image, width=None, height=None, inter=cv.INTER_AREA):
    newsize = (width, height)
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    # 高度算缩放比例
    if width is None:
        n = height / float(h)
        newsize = (int(n * w), height)
    else:
        n = width / float(w)
        newsize = (width, int(h * n))

    # 缩放图像
    newimage = cv.resize(image, newsize, interpolation=inter)
    return newimage


def on_mouse(event, x, y, flags, param):
    global img, img1, point1, point2, point1_dis, point2_dis
    img2 = img1.copy()
    if event == cv.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x*4, y*4)
        point1_dis = (x, y)
        cv.circle(img2, point1_dis, 10, (0,255,0), 5)
        cv.imshow('image', img2)
    elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv.rectangle(img2, point1_dis, (x, y), (255, 0, 0), 5)
        cv.imshow('image', img2)
    elif event == cv.EVENT_LBUTTONUP:         #左键释放
        point2 = (x*4, y*4)
        point2_dis = (x,y)
        cv.rectangle(img2, point1_dis, point2_dis, (0,0,255), 5)
        cv.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv.imwrite("output/%s/cutted.jpg"%(folder), cut_img)


def screenshots():
    global img, img1
    img = cv.imread("output/%s/input.jpg"%(folder))
    img1 = resizeImage(img, 816, 612)
    cv.namedWindow('image',1)
    cv.setMouseCallback('image', on_mouse)
    cv.imshow('image', img1)
    cv.waitKey(0)


def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


def wiener(input,PSF,eps,K=0.01):        #维纳滤波，K=0.01
    input_fft=fft.fft2(input)
    PSF_fft=fft.fft2(PSF) +eps
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    b = input_fft * PSF_fft_1
    result=fft.ifft2(b)
    result=np.abs(fft.fftshift(result))
    return result


def wiener_change(image):
    img_h = image.shape[0]
    img_w = image.shape[1]
    #graph.figure(0)
    #graph.xlabel("Original Image")
    #graph.gray()
    #graph.imshow(image)

    graph.figure(1)
    graph.gray()
    # 进行运动模糊处理
    PSF = motion_process((img_h, img_w), 60)

    out = wiener(image, PSF, winner_data)

    #graph.subplot(236)
    #graph.xlabel("wiener deblurred(k=0.01)")
    graph.imshow(out)
    graph.axis('off')
    graph.savefig('output/%s/winner_out.jpg'%(folder))
    graph.show()


def image_out(image, x, y, word):
    cv.namedWindow(word, 0)
    cv.resizeWindow(word, x, y)
    cv.imshow(word, image)


def contrast(image):
    dst = image
    img_h = image.shape[0]
    img_w = image.shape[1]
    graph.figure(1)
    graph.gray()
    # 进行运动模糊处理
    PSF = motion_process((img_h, img_w), 60)

    out = wiener(image, PSF, 1e-3)
    graph.imshow(out)
    graph.axis('off')
    graph.savefig('output/%s/winner_in.jpg'%(folder))
    graph.show()
    if contrast_data0 != 0:
        clache = cv.createCLAHE(clipLimit=contrast_data0, tileGridSize=(8, 8))
        dst = clache.apply(dst)
    if denosing_data != 0:
        dst = cv.fastNlMeansDenoising(dst,None ,denosing_data, 7, 21)
    if contrast_data1!=0:
        clache = cv.createCLAHE(clipLimit=contrast_data1, tileGridSize=(8, 8))
        dst = clache.apply(dst)
    if expansion_data != 0:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (expansion_data, expansion_data))
        dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)  # 开运算
    if rust_data != 0:
        kernel = np.ones((rust_data, rust_data), np.uint8)
        dst = cv.erode(dst, kernel)  # 腐蚀
    wiener_change(dst)
    return dst


def sharpen(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    dst = cv.filter2D(image , -1 , kernel=kernel)
    cv.namedWindow("median", 0)
    cv.resizeWindow("median", 600, 600)
    cv.imshow("median",dst)


def chang_contrast0(input):
    global contrast_data0
    contrast_data0 = float(input)


def chang_contrast1(input):
    global contrast_data1
    contrast_data1 = float(input)


def chang_denosing_data(input):
    global denosing_data
    denosing_data = float(input)


def change_expansion_data(input):
    global expansion_data
    expansion_data = int(input)


def change_rust_data(input):
    global rust_data
    rust_data = int(input)


def change_winner_data(input):
    global rust_data
    rust_data = float(input)


def scale_creat():
    s1 = tk.Scale(window,label='对比度0',from_=0.0 , to = 50.0,orient = tk.HORIZONTAL,
                  length = 250, showvalue = 1, tickinterval = 5, resolution = 1,command = chang_contrast0)
    s1.set(contrast_data0)
    s1.place(x= 300,y = 310)
    s2 = tk.Scale(window,label='对比度1',from_=0.0 , to = 2.5,orient = tk.HORIZONTAL,
                  length = 250, showvalue = 1, tickinterval = 0.5, resolution = 0.1,command = chang_contrast1)
    s2.set(contrast_data1)
    s2.place(x= 300,y = 390)
    s3 = tk.Scale(window,label='去噪',from_=0.0 , to = 20.0,orient = tk.HORIZONTAL,
                  length = 250, showvalue = 1, tickinterval = 5, resolution = 1,command = chang_denosing_data)
    s3.set(denosing_data)
    s3.place(x= 300,y = 470)
    s3 = tk.Scale(window,label='开运算系数',from_=0.0 , to = 50.0,orient = tk.HORIZONTAL,
                  length = 250, showvalue = 1, tickinterval = 10, resolution = 1,command = change_expansion_data)
    s3.set(expansion_data)
    s3.place(x= 300,y = 550)
    s4 = tk.Scale(window,label='腐蚀系数',from_=0.0 , to = 50.0,orient = tk.HORIZONTAL,
                  length = 250, showvalue = 1, tickinterval = 10, resolution = 1,command = change_rust_data)
    s4.set(rust_data)
    s4.place(x= 300,y = 630)
    b1 = tk.Button(window,text = '开始处理', width = 20,height = 3 , command = opencv)
    b1.place(x= 340,y = 710)


def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


def file_open():
    global a
    a = tkinter.filedialog.askopenfilename(filetypes=[("图片", ".jpg")])


def folder1():
    global folder
    folder = 1


def folder2():
    global folder
    folder = 2


def folder3():
    global folder
    folder = 3


def folder4():
    global folder
    folder = 4


def folder5():
    global folder
    folder = 5


def folder6():
    global folder
    folder = 6


def creat_bottom():
    top2 = tk.Toplevel()
    top2.title = ('设定存储文件夹')
    top2.geometry('400x220')
    r1 = tk.Button(top2, text='1',width = 10,
                        command = folder1)
    r1.pack()
    r2 = tk.Button(top2, text='2',width = 10,
                        command = folder2)
    r2.pack()
    r3 = tk.Button(top2, text='3',width = 10,
                        command = folder3)
    r3.pack()
    r4 = tk.Button(top2, text='4',width = 10,
                        command = folder4)
    r4.pack()
    r5 = tk.Button(top2, text='5',width = 10,
                        command = folder5)
    r5.pack()
    r6 = tk.Button(top2, text='test',width = 10,
                        command = folder6)
    r6.pack()
    r7 = tk.Button(top2, text='确认',width = 20,
                        command = top2.destroy)
    r7.pack()
    top2.mainloop()


def creat_menu():
    menubar = tk.Menu(window)
    filemenu = tk.Menu(menubar, tearoff = 0)
    menubar.add_cascade(label='文件', menu=filemenu)
    helpmenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='帮助', menu=helpmenu)
    filemenu.add_cascade(label='选择文件夹', command=creat_bottom)
    filemenu.add_cascade(label='摄像头',command = photograph)
    filemenu.add_cascade(label='切割图像', command=screenshots)
    helpmenu.add_cascade(label='关于',command = about_creat)
    window.config(menu=menubar)


def exposure_change(input):
    global exposure
    exposure = input


def logic_change(input):
    global logic
    logic = input


def photograph():
    global address0
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, 1196444237)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 3264)  # 设置分辨率
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2448)

    #cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC, cv.CV_FOURCC('M', 'J', 'P', 'G'))
    #844715353
    #CAP_PROP_FOURCC
    #1196444237
    #cv.VideoWriter_fourcc(*'MJPG)


    cv.namedWindow("摄像头", 0)
    cv.resizeWindow("摄像头", 800, 600)
    cv.createTrackbar("更改曝光","摄像头", 0, 15, exposure_change)
    switch = '0:OFF\n1:ON'
    cv.createTrackbar(switch, '摄像头', 0, 1, logic_change)

    cap.set(cv.CAP_PROP_FOURCC,cv.COLOR_YUV2BGR_YUY2)
    while (1):
        # get a frame
        if logic == 0:
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE,logic)
            cap.set(cv.CAP_PROP_EXPOSURE,exposure-15)
        ret, frame = cap.read()
        # show a frame
        cv.imshow("摄像头", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.imwrite("output/%s/input.jpg"%(folder), frame)
            address0 = "output/%s/input.jpg"%(folder)

            cavans_creat()
            break
        elif cv.waitKey(1) & 0xFF == ord('c'):
            break
    cap.release()
    cv.destroyAllWindows()


def cavans_creat():
    global photo0
    global photo1
    #address0 = "output/%s/cutted.jpg" % (folder)
    img0 = Image.open(address0)
    img0 = resize(3264, 2448, 400, 300, img0)
    photo0 = ImageTk.PhotoImage(img0)  # 用PIL模块的PhotoImage打开
    img1 = Image.open(address1)
    img1 = resize(3264, 2448, 400, 300, img1)
    photo1 = ImageTk.PhotoImage(img1)  # 用PIL模块的PhotoImage打开
    canvas0 = tk.Canvas(window, bg ='white',height=300,width=400)
    canvas0.create_image(0,0,anchor = 'nw',image = photo0)
    canvas0.place(x= 0, y= 0)
    canvas1 = tk.Canvas(window, bg ='white',height=300,width=400)
    canvas1.create_image(0,0,anchor = 'nw',image = photo1)
    canvas1.place(x= 410, y= 0)


def about_creat():
    top1=tk.Toplevel()
    top1.title('关于本程序')
    top1.geometry('300x200')
    image = Image.open('code_image\\111.jpg')
    img = ImageTk.PhotoImage(image)
    word_box = tk.Label(top1, text='毛玻璃清晰化处理软件\r版本：1.7\r编写者：张逸航')
    canvas1 = tk.Canvas(top1, width = 80 ,height = 80, bg = 'white')
    canvas1.create_image(0,0,image = img,anchor="nw")
    canvas1.create_image(image.width,0,image = img,anchor="nw")
    canvas1.pack()
    word_box.pack()
    top1.mainloop()


def opencv():
    global address1
    src = cv.imread('output/%s/cutted.jpg'%(folder))
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite('output/%s/gray.jpg' % (folder),src)
    #wiener_change(src)

    #image_out(src, 600, 800, "input_image")
    out = contrast(src)
    #image_out(out, 600, 600, "out")
    cv.imwrite('output/%s/output1.jpg'%(folder),out)
    address1 = 'output/%s/output1.jpg' % (folder)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cavans_creat()


creat_menu()
cavans_creat()
scale_creat()
window.mainloop()