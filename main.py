import sys
import os
import cv2 
import numpy as np
import torchvision
import torch
import pandas as pd  
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from PyQt5.QtWidgets import ( QApplication, QWidget, QLabel, QPushButton,QTextEdit, QComboBox, QFileDialog,QVBoxLayout, QHBoxLayout, QMessageBox )
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
except Exception:
    fasterrcnn_resnet50_fpn_v2 = None
try:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
except Exception:
    fasterrcnn_mobilenet_v3_large_fpn = None
try:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
except Exception:
    fasterrcnn_mobilenet_v3_large_320_fpn = None
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead



class ColonyApp(QWidget):                                   #定义一个ColonyApp的类，QWidget是 PyQt5 的基础窗口类，在此基础上进行GUI设计
    def __init__(self):                                     #__init__ 是类的构造函数，当创建对象时自动执行
        super().__init__()                                  #调用父类 QWidget 的构造函数，进行初始化
        self.setWindowTitle("CCI菌落识别工具")               #窗口名
        self.resize(1600, 1200)                             #窗口大小（像素）

        # 初始化变量
        self.current_image_path = None                      #上传图像的路径
        self.models = {}                                    #选用模型的字典，键为名称值为模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.original_pixmap = None                         #存储原始图片的 QPixmap 对象
        self.annotated_pixmap = None                        #存储标注图片的 QPixmap 对象
        self.is_showing_annotated = False                   #当前图片是原始/标注
        self.model_info = {}                                #模型信息
        
        # 初始化UI
        self.init_ui()

        # 加载模型
        self.load_models()

    def _get_fasterrcnn_builder(self, arch_name):
        builders = {
            "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
            "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
            "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
        }
        return builders.get(arch_name)

    def _load_fasterrcnn_state(self, model, model_path):
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        return missing, unexpected

    def _build_fasterrcnn_manual(self, info):
        """
        Build FasterRCNN exactly like your training code (manual assembly).
        Required in info:
        - num_classes
        - anchor_sizes_per_level (optional; defaults to your training values)
        - anchor_ratios (optional)
        - min_size / max_size (optional)
        - trainable_backbone_layers (optional, not important for inference)
        """
        num_classes = info["num_classes"]
        min_size = info.get("min_size", 960)
        max_size = info.get("max_size", 1536)

        anchor_sizes_per_level = info.get(
            "anchor_sizes_per_level",
            ((8,), (16,), (32,), (64,), (128,))   # 和你训练代码一致
        )
        anchor_ratios = info.get("anchor_ratios", (0.5, 1.0, 2.0))

        # backbone: ResNet50 + FPN
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=ResNet50_Weights.DEFAULT,  # 推理时有无都行，但不影响加载 state_dict（不冲突）
            trainable_layers=info.get("trainable_backbone_layers", 5),
        )

        # anchors
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes_per_level,
            aspect_ratios=(anchor_ratios,) * len(anchor_sizes_per_level),
        )

        # rpn head必须匹配 anchors per location
        rpn_head = RPNHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            conv_depth=2,
        )

        model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_train=info.get("rpn_pre_nms_top_n_train", 4000),
            rpn_pre_nms_top_n_test=info.get("rpn_pre_nms_top_n_test", 2000),
            rpn_post_nms_top_n_train=info.get("rpn_post_nms_top_n_train", 2000),
            rpn_post_nms_top_n_test=info.get("rpn_post_nms_top_n_test", 1000),
            rpn_fg_iou_thresh=info.get("rpn_fg_iou_thresh", 0.7),
            rpn_bg_iou_thresh=info.get("rpn_bg_iou_thresh", 0.3),
            box_score_thresh=info.get("box_score_thresh", 0.0),
            box_nms_thresh=info.get("box_nms_thresh", 0.5),
            box_detections_per_img=info.get("box_detections_per_img", 200),
        )
        return model

    def _build_fasterrcnn(self, info):
        
        if info.get("arch") == "manual_resnet50_fpn":
            return self._build_fasterrcnn_manual(info)
        
        num_classes = info["num_classes"]
        arch = info.get("arch", "fasterrcnn_resnet50_fpn")
        min_size = info.get("min_size", 800)
        max_size = info.get("max_size", 1333)

        builder = self._get_fasterrcnn_builder(arch)
        if builder is None:
            raise ValueError(f"未知 Faster R-CNN 架构: {arch}")

        model = builder(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _frcnn_label_to_name(self, model_name, label_id):
        info = self.model_info.get(model_name, {})
        names = info.get("class_names")
        if names and label_id < len(names):
            return names[label_id]
        return f"class_{label_id}"

    def init_ui(self):                                      #定义类方法 init_ui，用于初始化 GUI 界面。self 可以访问和修改类的实例变量。

        # 左侧布局 
        left_layout = QVBoxLayout()                         #创建一个垂直布局，上下插入控件

        # 1. 图片展示区域
        self.image_label = QLabel("请上传图片以进行识别")     #创建一个标签控件
        self.image_label.setAlignment(Qt.AlignCenter)       #居中
        self.image_label.setFixedSize(1200, 1000)           #大小（不可缩放）
        self.image_label.setStyleSheet("border: 2px dashed #aaa; color: #aaa; font-size: 20px;")    #边框文本参数
        left_layout.addWidget(self.image_label)             #加入控件

        # 2. 上传/批量处理按钮区域
        button_container_layout = QHBoxLayout()             #创建一个水平布局放按钮
        upload_btn = QPushButton("上传图片")                 #创建一个按钮
        upload_btn.clicked.connect(self.upload_image)       #点击按钮触发self.upload_image函数
        batch_btn = QPushButton("批量处理")         #创建另一个按钮
        batch_btn.clicked.connect(self.batch_process)       #点击按钮触发self.batch_process函数

        button_container_layout.addWidget(upload_btn)       #水平布局加入按钮
        button_container_layout.addWidget(batch_btn)
        left_layout.addLayout(button_container_layout)      #将水平布局加到垂直布局中

        # ---------- 右侧布局 ----------
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)                         #控件间距15像素

        # 3. 模型选择下拉框
        right_layout.addWidget(QLabel("选择识别模型:"))      
        self.model_box = QComboBox()                        #下拉框控件
        self.model_box.setFixedHeight(40)                   #固定高度
        right_layout.addWidget(self.model_box)

        # 4. 开始识别按钮
        self.start_btn = QPushButton("开始识别")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("font-size: 18px; background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_detection)
        right_layout.addWidget(self.start_btn)

        # 5. 新增：切换视图按钮
        self.toggle_btn = QPushButton("切换原始/标注图")
        self.toggle_btn.setFixedHeight(40)
        self.toggle_btn.clicked.connect(self.toggle_image_view)
        self.toggle_btn.setEnabled(False)                   # 初始状切换功能为禁用
        right_layout.addWidget(self.toggle_btn)

        # 6. 结果输出文本框
        right_layout.addWidget(QLabel("识别结果:"))
        self.result_box = QTextEdit()
        self.result_box.setPlaceholderText("这里将显示识别结果的统计信息...")
        self.result_box.setReadOnly(True)                   #文本框只读
        right_layout.addWidget(self.result_box)

        # ---------- 主布局 ----------
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)               #比例左三右一
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)   
                              #使用布局

    def load_models(self):
        """在程序启动时加载所有模型"""
        self.result_box.append("正在加载模型，请稍候...")
        self.result_box.append(f"当前使用的计算设备: {self.device}")
        QApplication.processEvents()

        self.model_info = {
            "YOLOv8-basic": {
                "path": r"E:\CCI-Colony Counting and Identification\CCI\model\YOLO\YOLOv8n_base.pt",
                "type": "yolo"
            },
            "YOLOv8-augmented": {
                "path": r"E:\CCI-Colony Counting and Identification\CCI\model\YOLO\YOLOv8n_aug.pt",
                "type": "yolo"
            },
            "YOLOv8-final": {
                "path": r"E:\CCI-Colony Counting and Identification\CCI\model\YOLO\YOLOv8m_final.pt",
                "type": "yolo"
            },
             "FasterRCNN-baseline": {
                 "path": r"E:\CCI-Colony Counting and Identification\CCI\model\FasterRCNN\base.pth",
                 "type": "fasterrcnn",
                 "num_classes": 12,  # 包含背景类
                 "arch": "fasterrcnn_resnet50_fpn",
                 # 可选：如果训练时改过输入尺寸
                 # "min_size": 800,
                 # "max_size": 1333,
                 # 可选：类别名列表，索引0为背景"Alternaria alternata", 1: "Alternaria tenuissima", 2: "Bacillus subtilis", 3: "Bacillus thaonhiensis", 4: "Deinococcus soli", 5: "Kocuria oceani", 6: "Arthrobacter oryzae", 7: "Micrococcus luteus", 8: "Staphylococcus aureus", 9: "Streptomyces spororaveus"}
                  "class_names": ["__background__", "Alternaria alternata", "Alternaria tenuissima", "Bacillus subtilis", "Bacillus thaonhiensis", "Deinococcus soli", "Kocuria oceani", "Arthrobacter oryzae", "Micrococcus luteus", "Staphylococcus aureus", "Streptomyces spororaveus"],
             },
             "FasterRCNN-gen": {
                 "path": r"E:\CCI-Colony Counting and Identification\CCI\model\FasterRCNN\synthesis.pth",
                 "type": "fasterrcnn",
                 "num_classes": 12,  # 包含背景类
                 "arch": "fasterrcnn_resnet50_fpn",
                 # 可选：如果训练时改过输入尺寸
                 # "min_size": 800,
                 # "max_size": 1333,
                 # 可选：类别名列表，索引0为背景
                  "class_names": ["__background__", "Alternaria alternata", "Alternaria tenuissima", "Bacillus subtilis", "Bacillus thaonhiensis", "Deinococcus soli", "Kocuria oceani", "Arthrobacter oryzae", "Micrococcus luteus", "Staphylococcus aureus", "Streptomyces spororaveus"],
             },
             #如果有更多模型，在这里添加
             "FasterRCNN-manual": {
                 "path": r"E:\CCI-Colony Counting and Identification\CCI\model\FasterRCNN\manual_final.pth",
                 "type": "fasterrcnn",
                 "num_classes": 12,
                 "arch": "manual_resnet50_fpn",

                # 和训练一致（强烈建议写上，避免忘）
                 "min_size": 960,
                 "max_size": 1536,
                 "anchor_sizes_per_level": ((8,), (16,), (32,), (64,), (128,)),
                 "anchor_ratios": (0.5, 1.0, 2.0),

                 "class_names": ["__background__", "Alternaria alternata", "Alternaria tenuissima",
                                "Bacillus subtilis", "Bacillus thaonhiensis", "Deinococcus soli",
                                "Kocuria oceani", "Arthrobacter oryzae", "Micrococcus luteus",
                                "Staphylococcus aureus", "Streptomyces spororaveus"],
            },
        }
        
        # 遍历所有待加载的模型
        for model_name, info in self.model_info.items():
            model_path = info["path"]
            model_type = info["type"]

            if not os.path.exists(model_path):
                self.result_box.append(f"警告：找不到模型文件 {model_path}，已跳过 {model_name}。")
                continue

            try:
                model = None                 
                if model_type == "yolo":
                    model = YOLO(model_path)

                elif model_type == "fasterrcnn":
                    model = self._build_fasterrcnn(info)
                    missing, unexpected = self._load_fasterrcnn_state(model, model_path)
                    model.eval()
                    model.roi_heads.detections_per_img = 300
                    model.to(self.device)

                    if missing or unexpected:
                        self.result_box.append(
                            f"提示: {model_name} 权重与模型结构不完全匹配 "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})"
                        )

                else:
                    self.result_box.append(f"错误：未知的模型类型 {model_type}，无法加载 {model_name}。")
                    continue

                if model:
                    self.models[model_name] = model         #建立与窗口关联的模型集（下拉框、存储）
                    self.model_box.addItem(model_name)
                    self.result_box.append(f" {model_name} 加载成功！")

            except Exception as e:
                self.result_box.append(f"加载模型 {model_name} 时发生错误: {e}")


    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")   #文件选择对话框，返回路径，参数为父窗口，标题，默认路径，类型
        if file:
            self.current_image_path = file
            self.result_box.append(f"已选择图片: {file}")
            
            # 保存原始图像的Pixmap
            self.original_pixmap = QPixmap(file)            #暂存图片
            
            # 重置状态
            self.annotated_pixmap = None                    #清空之前标注
            self.toggle_btn.setEnabled(False)               #禁用切换
            self.toggle_btn.setText("切换原始/标注图")         
            self.is_showing_annotated = False               #显示原图

            # 在标签上显示原始图片
            scaled_pixmap = self.original_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)    #缩放
            self.image_label.setPixmap(scaled_pixmap)       #左侧图片显示
            self.image_label.setStyleSheet("border: 1px solid black;")      #边框

    def batch_process(self):

        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        
        if not folder:
            return  

        self.result_box.append(f"\n开始批量处理文件夹: {folder}")
        QApplication.processEvents()

        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = []                                             #存储图片名
        try:
            for filename in os.listdir(folder):                      #列出所有子名称
                file_path = os.path.join(folder, filename)           #得到路径
                if os.path.isfile(file_path):                        #如果是文件
                    ext = os.path.splitext(filename)[1].lower()      #获取小写扩展名
                    if ext in allowed_extensions:                    #是图片
                        image_files.append(filename)
                    else:
                        error_msg = f"文件夹中包含非图片文件: '{filename}'。\n请确保文件夹内只包含图片文件。"
                        self.result_box.append(f"错误: {error_msg}")
                        QMessageBox.warning(self, "处理中止", error_msg)
                        return
        except Exception as e:
            error_msg = f"读取文件夹失败: {e}"
            self.result_box.append(f"错误: {error_msg}")
            QMessageBox.critical(self, "错误", error_msg)
            return

        if not image_files:
            info_msg = "所选文件夹中未找到任何图片文件。"
            self.result_box.append(info_msg)
            QMessageBox.information(self, "提示", info_msg)
            return

        if not self.models:
            error_msg = "错误：没有可用的模型，请检查模型路径！"
            self.result_box.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            return

        model_name = self.model_box.currentText()
        model = self.models[model_name]
        model_type = self.model_info[model_name]["type"]

        self.result_box.append(f"使用模型 '{model_name}' 进行批量识别...")
        QApplication.processEvents()

        # 3. 循环处理图片并收集结果
        results_data = {}
        all_class_names = set()
        # 类别名称映射字典
        yolo_id_to_name = {0: "Alternaria alternata", 1: "Alternaria tenuissima", 2: "Bacillus subtilis", 3: "Bacillus thaonhiensis", 4: "Deinococcus soli", 5: "Kocuria oceani", 6: "Arthrobacter oryzae", 7: "Micrococcus luteus", 8: "Staphylococcus aureus", 9: "Streptomyces spororaveus"}
        confidence_threshold = 0.5

        try:
            for i, filename in enumerate(image_files):      #索引遍历
                self.result_box.append(f"({i+1}/{len(image_files)}) 正在处理: {filename}")
                QApplication.processEvents()
                full_path = os.path.join(folder, filename)
                counts = {}

                # --- 根据模型类型选择不同的处理路径 ---
                if model_type == "yolo":
                    results = model(full_path, verbose=False)
                    result = results[0]
                    if result.boxes:
                        for cls_id in result.boxes.cls.tolist():               #结果的框的类别转化为列表 
                            cls_id = int(cls_id)
                            class_name = yolo_id_to_name.get(cls_id, f"未知_{cls_id}")
                            counts[class_name] = counts.get(class_name, 0) + 1
                
                elif model_type == "fasterrcnn":
                    image_bgr = cv2.imread(full_path)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    img_tensor = F.to_tensor(pil_image).to(self.device)        #fasterRCNN输入格式转换

                    with torch.no_grad():
                        predictions = model([img_tensor])                      #禁用梯度计算
                    
                    preds = predictions[0]                                     #检测结果
                    scores = preds['scores'].cpu().numpy()                     #转化为numpy
                    labels = preds['labels'].cpu().numpy()
                    
                    filtered_indices = scores > confidence_threshold           #一列表的false和true
                    filtered_labels = labels[filtered_indices]                 #仅保留高于阈值的结果

                    if len(filtered_labels) > 0:
                        for label_id in filtered_labels:
                            class_name = self._frcnn_label_to_name(model_name, label_id)
                            counts[class_name] = counts.get(class_name, 0) + 1
                
                all_class_names.update(counts.keys())                          #记录所有出现过的类别
                results_data[filename] = counts

        except Exception as e:
            error_msg = f"处理文件 '{filename}' 时发生错误: {e}"
            self.result_box.append(f"错误: {error_msg}")
            QMessageBox.critical(self, "处理错误", error_msg)
            return

        # 4. 生成并保存表格文件
        try:
            sorted_class_names = sorted(list(all_class_names))                 #类别转化为列表并排序
            df = pd.DataFrame.from_dict(results_data, orient='index', columns=sorted_class_names)     #构建dataframe，不管个例图片有无全部种类成列
            df = df.fillna(0).astype(int)                                      #缺值补0

            if not df.empty:
                total_row = df.sum().to_frame('合计').T
                df = pd.concat([df, total_row])                                #对列求和，加在底部

            df.reset_index(inplace=True)                                       #重置索引，成为第一列（原先是index）
            df = df.rename(columns={'index': '图片文件名'})                     #修改列名

            output_path = os.path.join(folder, "batch_results.csv")
            df.to_csv(output_path, index=False, encoding='utf-8-sig')

            self.result_box.append("\n------ 批量处理完成 ------")
            self.result_box.append(f"结果已保存至: {output_path}")
            QMessageBox.information(self, "完成", f"批量处理已完成！\n结果已保存至:\n{output_path}")

        except Exception as e:
            error_msg = f"保存结果文件时发生错误: {e}"
            self.result_box.append(f"错误: {error_msg}")
            QMessageBox.critical(self, "保存失败", error_msg)

    def toggle_image_view(self):
        """在原始图像和标注图像之间切换"""
        if not self.original_pixmap or not self.annotated_pixmap:
            return

        if self.is_showing_annotated:
            # 当前显示的是标注图，切换到原始图
            pixmap_to_show = self.original_pixmap
            self.toggle_btn.setText("查看标注图")
        else:
            # 当前显示的是原始图，切换到标注图
            pixmap_to_show = self.annotated_pixmap
            self.toggle_btn.setText("查看原始图")

        scaled_pixmap = pixmap_to_show.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.is_showing_annotated = not self.is_showing_annotated

    def start_detection(self):
        """开始识别"""
        if not self.current_image_path:
            self.result_box.append("错误：请先上传一张图片！")
            return

        if not self.models:
            self.result_box.append("错误：没有可用的模型，请检查模型路径！")
            return

        model_name = self.model_box.currentText()            #获取用户在下拉框选择的模型名称。      
        model = self.models[model_name]
        model_type = self.model_info[model_name]["type"]     #字典中取出对应的模型

        self.result_box.append(f"\n使用模型 '{model_name}' 进行识别...")
        QApplication.processEvents()

        try:
            if model_type == "yolo":
                # YOLO模型的处理逻辑
                results = model(self.current_image_path)
                result = results[0]

                # 获取带标注的图片
                annotated_image_bgr = result.plot(labels=False, conf=True, line_width=2)
                
                # 统计结果
                counts = {}
                if result.boxes:
                    names = model.names
                    for cls_id in result.boxes.cls.tolist():
                        cls_id = int(cls_id)
                        counts[cls_id] = counts.get(cls_id, 0) + 1
                else:
                    self.result_box.append("未检测到任何目标。")
                id_to_name = {0: "Alternaria alternata", 1: "Alternaria tenuissima", 2: "Bacillus subtilis", 3: "Bacillus thaonhiensis", 4: "Deinococcus soli", 5: "Kocuria oceani", 6: "Arthrobacter oryzae", 7: "Micrococcus luteus", 8: "Staphylococcus aureus", 9: "Streptomyces spororaveus"}
                counts = {id_to_name[k]: v for k, v in counts.items()}

            elif model_type == "fasterrcnn":
                # Faster R-CNN模型的处理逻辑
                # 预处理图片：将OpenCV图像转换为PIL图像，再转换为张量
                image_bgr = cv2.imread(self.current_image_path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                img_tensor = F.to_tensor(pil_image).to(self.device)

                # 执行推理
                with torch.no_grad():
                    predictions = model([img_tensor])
                
                # Faster R-CNN的输出是字典，包含 'boxes', 'labels', 'scores'
                preds = predictions[0]
                boxes = preds['boxes'].cpu().numpy()
                labels = preds['labels'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()

                # 筛选置信度高于阈值的检测结果 (例如，0.5)
                # 你可以添加一个配置项来控制这个阈值
                confidence_threshold = 0.5
                filtered_indices = scores > confidence_threshold
                filtered_boxes = boxes[filtered_indices]
                filtered_labels = labels[filtered_indices]
                
                # 绘制边界框和标签
                annotated_pil_image = pil_image.copy()
                draw = ImageDraw.Draw(annotated_pil_image)

                # 你需要一个类别名称列表来映射标签ID
                # 类别名称由 model_info 中的 class_names 提供（索引 0 为背景）

                counts = {}
                if len(filtered_boxes) > 0:
                    for i in range(len(filtered_boxes)):
                        box = filtered_boxes[i]
                        label = filtered_labels[i]
                        score = scores[filtered_indices][i]
                        
                        # 绘制矩形框
                        draw.rectangle(box, outline='red', width=3)
                        
                        # 绘制类别和置信度文本
                        label_name = self._frcnn_label_to_name(model_name, label)
                        text = f"{label_name}: {score:.2f}"
                        draw.text((box[0], box[1] - 15), text, fill='red')

                        # 统计类别数量
                        counts[label_name] = counts.get(label_name, 0) + 1
                else:
                    self.result_box.append("未检测到任何目标。")

                # 将 PIL 图像转换为 OpenCV 格式以便后续处理
                annotated_image_bgr = cv2.cvtColor(np.array(annotated_pil_image), cv2.COLOR_RGB2BGR)

            else:
                self.result_box.append("错误：未知的模型类型，无法进行识别。")
                return
                
            # --- 通用处理部分 ---
            
            # 将OpenCV图像转换为PyQt的QPixmap并保存
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
            height, width, channel = annotated_image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.annotated_pixmap = QPixmap.fromImage(q_image)
            
            # 默认显示标注图
            scaled_pixmap = self.annotated_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # 更新切换按钮的状态
            self.toggle_btn.setEnabled(True)
            self.toggle_btn.setText("查看原始图")
            self.is_showing_annotated = True

            # 统计结果并显示
            self.result_box.append(" ")
            self.result_box.append("------ 识别结果统计 ------")
            self.result_box.append(" ")
            if counts:
                for class_name, count in counts.items():
                    self.result_box.append(f"'{class_name}': {count} ")
            else:
                self.result_box.append("未检测到任何目标。")

        except Exception as e:
            self.result_box.append(f"识别过程中发生错误: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)                            #创建一个 QApplication 实例，是所有必须有的对象
    window = ColonyApp()
    window.show()
    sys.exit(app.exec_())                                   #app.exec_()启动 Qt 事件循环，关闭后返回整数退出

