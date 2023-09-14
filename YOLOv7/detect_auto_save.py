import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect():

    #获取，输入输出文件夹源，权重，参数等参数
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #地址
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:#开启摄像头
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    """
    path 图片/视频路径
    img 进行resize+pad之后的图片
    img0 原size图片
    vid_cap 当读取图片时为None，读取视频时为视频源
    """
    sumframe = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS过滤
        #NMS是一种让你确保算法只对每个对象得到一个检测框的方法。
        """
               pred:前向传播的输出
               conf_thres:置信度阈值
               iou_thres:iou阈值
               classes:是否只保留特定的类别
               agnostic:进行nms是否也去除不同类别之间的框
               经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
               pred是一个列表list[torch.tensor]，长度为batch_size
               每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
         """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        #检测成功
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections 处理检测框
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path bus.jpg
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        #标置信度 face：0.88
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0) #函数可以在窗口中显示图像
                cv2.waitKey(1)  # 1 millisecond

            """"处理mp4视频文件"""
            strmp4 = "mp4"
            if (strmp4 in str(p.name)):
                fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 帧速率/帧数/fps
                """每0.5s一截图"""
                frameRate = int(0.5 * int(fps))  # 帧数间隔
                print(sumframe % frameRate)
                if (sumframe % frameRate == 0):
                    for j, detection in enumerate(det):
                        # i检测框数量
                        x, y, x1, y1 = detection[0:4]
                        x = int(x.item())
                        y = int(y.item())
                        x1 = int(x1.item())
                        y1 = int(y1.item())
                        crop_img = im0[y:y1, x:x1]  # [从上到下，从左到右]
                        filepath = str(save_dir) + '\cut-' + str(p.name) + '\Frame' + str(sumframe) # save_dir=./……/exp
                        if not os.path.exists(filepath):  # 如果路径不存在
                            os.makedirs(filepath)
                        filename = filepath  + '\Face' + str(j + 1) + '.jpg'  # 取名
                        cv2.imwrite(filename, crop_img)
                        print('Saved detection:', filename)


            # Save results (image with detections)
            if save_img:
                #图片
                if dataset.mode == 'image':
                    """循环遍历每个检测框，并将其保存为图像文件"""
                    for j, detection in enumerate(det):
                        # i检测框数量
                        x, y, x1, y1 = detection[0:4]
                        x = int(x.item())
                        y = int(y.item())
                        x1 = int(x1.item())
                        y1 = int(y1.item())
                        crop_img = im0[y:y1, x:x1]  # [从上到下，从左到右]
                        filepath = str(save_dir) + '\cut-' + str(p.name)  # save_dir=./……/exp
                        if not os.path.exists(filepath):  # 如果路径不存在
                            os.makedirs(filepath)
                        filename = filepath + '\Face' + str(j + 1) + '.jpg'  # 取名
                        cv2.imwrite(filename, crop_img)
                        print('Saved detection:', filename)

                    cv2.imwrite(save_path, im0)#（图片名，处理后图片）
                    print(f" The image with the result is saved in: {save_path}")

                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        #来判断一个对象是否是一个已知的类型(如果对象的类型与参数二的类型相同则返回 True，否则返回 False)
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) #帧速率/帧数/fps
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #在视频流的帧的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #在视频流的帧的高度

                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        # 保存的文件的路径,指定编码器,要保存的视频的帧率,要保存的文件的画面尺寸
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    vid_writer.write(im0)
                sumframe = sumframe + 1


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
        """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='Face_Detection.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/Droneface_test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():

        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['face.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


