from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F
from ultralytics import YOLO
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

try:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
except Exception:
    fasterrcnn_mobilenet_v3_large_320_fpn = None

try:
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
except Exception:
    fasterrcnn_mobilenet_v3_large_fpn = None

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
except Exception:
    fasterrcnn_resnet50_fpn_v2 = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_MASTER_CLASS_NAMES = [
    "Alternaria alternata",
    "Alternaria tenuissima",
    "Bacillus subtilis",
    "Bacillus thaonhiensis",
    "Deinococcus soli",
    "Kocuria oceani",
    "Arthrobacter oryzae",
    "Micrococcus luteus",
    "Staphylococcus aureus",
    "Streptomyces spororaveus",
]


def normalize_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def build_name_to_index(class_names):
    return {normalize_name(name): idx for idx, name in enumerate(class_names)}


def default_model_specs(root: str | Path, class_names: list[str]):
    root = Path(root)
    frcnn_names = ["__background__"] + list(class_names)
    num_classes = len(class_names) + 1
    return [
        {
            "enabled": True,
            "name": "YOLOv8-basic",
            "type": "yolo",
            "weights": root / "model" / "YOLO" / "YOLOv8n_base.pt",
        },
        {
            "enabled": False,
            "name": "YOLOv8-augmented",
            "type": "yolo",
            "weights": root / "model" / "YOLO" / "YOLOv8n_aug.pt",
        },
        {
            "enabled": True,
            "name": "YOLOv8-final",
            "type": "yolo",
            "weights": root / "model" / "YOLO" / "YOLOv8m_final.pt",
        },
        {
            "enabled": True,
            "name": "FasterRCNN-baseline",
            "type": "fasterrcnn",
            "weights": root / "model" / "FasterRCNN" / "base.pth",
            "arch": "fasterrcnn_resnet50_fpn",
            "num_classes": num_classes,
            "min_size": 800,
            "max_size": 1333,
            "class_names_with_background": frcnn_names,
        },
        {
            "enabled": False,
            "name": "FasterRCNN-gen",
            "type": "fasterrcnn",
            "weights": root / "model" / "FasterRCNN" / "synthesis.pth",
            "arch": "fasterrcnn_resnet50_fpn",
            "num_classes": num_classes,
            "min_size": 800,
            "max_size": 1333,
            "class_names_with_background": frcnn_names,
        },
        {
            "enabled": True,
            "name": "FasterRCNN-manual",
            "type": "fasterrcnn",
            "weights": root / "model" / "FasterRCNN" / "manual_final.pth",
            "arch": "manual_resnet50_fpn",
            "num_classes": num_classes,
            "min_size": 960,
            "max_size": 1536,
            "anchor_sizes_per_level": ((8,), (16,), (32,), (64,), (128,)),
            "anchor_ratios": (0.5, 1.0, 2.0),
            "class_names_with_background": frcnn_names,
        },
    ]


def make_default_config(root: str | Path):
    root = Path(root)
    return {
        "root": root,
        "output_dir": root / "draw" / "eval_results",
        "test_images_dir": "",
        "gt_source": "coco",
        "coco_ann_path": "",
        "yolo_labels_dir": "",
        "master_class_names": DEFAULT_MASTER_CLASS_NAMES.copy(),
        "yolo_gt_class_names": DEFAULT_MASTER_CLASS_NAMES.copy(),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "image_suffixes": IMAGE_SUFFIXES,
        "map_iou_thresholds": np.arange(0.50, 0.96, 0.05),
        "confusion_iou": 0.50,
        "pr_curve_iou": 0.50,
        "confusion_conf": 0.25,
        "count_conf": 0.25,
        "strict_gt_category_check": False,
        "raw_pred_conf": 0.001,
        "nms_iou": 0.70,
        "max_det": 500,
        "threshold_sweep": np.linspace(0.05, 0.95, 19),
        "model_specs": default_model_specs(root, DEFAULT_MASTER_CLASS_NAMES),
    }


def validate_config(cfg):
    if not cfg.get("test_images_dir"):
        raise ValueError("请先填写 test_images_dir。")

    image_dir = Path(cfg["test_images_dir"])
    if not image_dir.exists():
        raise FileNotFoundError(f"测试集图片目录不存在: {image_dir}")

    gt_source = str(cfg.get("gt_source", "")).strip().lower()
    if gt_source not in {"coco", "yolo"}:
        raise ValueError("gt_source 只能是 'coco' 或 'yolo'。")

    if gt_source == "coco" and not cfg.get("coco_ann_path"):
        raise ValueError("gt_source='coco' 时必须填写 coco_ann_path。")
    if gt_source == "yolo" and not cfg.get("yolo_labels_dir"):
        raise ValueError("gt_source='yolo' 时必须填写 yolo_labels_dir。")

    enabled_models = [spec for spec in cfg["model_specs"] if spec.get("enabled", True)]
    if not enabled_models:
        raise ValueError("至少启用一个模型。")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, enabled_models


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def yolo_to_xyxy(box, width, height):
    x_center, y_center, w, h = box
    x1 = (x_center - w / 2.0) * width
    y1 = (y_center - h / 2.0) * height
    x2 = (x_center + w / 2.0) * width
    y2 = (y_center + h / 2.0) * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def clip_boxes_xyxy(boxes, width, height):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def box_iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-9, a_max=None)


def make_empty_annotations():
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


def filter_predictions(preds, conf_threshold):
    keep = preds["scores"] >= conf_threshold
    return {
        "boxes": preds["boxes"][keep],
        "labels": preds["labels"][keep],
        "scores": preds["scores"][keep],
    }


def compute_ap(recall, precision):
    if len(recall) == 0 or len(precision) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_missing_classes(model_class_names, master_class_names):
    model_set = {
        normalize_name(name)
        for name in model_class_names
        if normalize_name(name) not in {"__background__", "background", "bg"}
    }
    master_set = {normalize_name(name) for name in master_class_names}
    missing = [name for name in master_class_names if normalize_name(name) not in model_set]
    extra = [
        name
        for name in model_class_names
        if normalize_name(name) not in master_set and normalize_name(name) not in {"__background__", "background", "bg"}
    ]
    return missing, extra


def load_ground_truth_from_coco(images_dir, coco_json_path, master_class_names, strict_category_check=False):
    images_dir = Path(images_dir)
    coco_json_path = Path(coco_json_path)
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    name_to_index = build_name_to_index(master_class_names)
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    used_category_ids = {
        ann["category_id"]
        for ann in coco.get("annotations", [])
        if not ann.get("iscrowd", 0)
    }
    cat_id_to_master = {}
    ignored_category_names = []
    for cat_id, cat_name in categories.items():
        key = normalize_name(cat_name)
        if key not in name_to_index:
            if strict_category_check and cat_id in used_category_ids:
                raise KeyError(f"COCO category 未在 master_class_names 中找到: {cat_name}")
            ignored_category_names.append(cat_name)
            continue
        cat_id_to_master[cat_id] = name_to_index[key]

    if ignored_category_names:
        unique_ignored = sorted(set(ignored_category_names))
        print(f"忽略不参与评估的 COCO 类别: {unique_ignored}")

    gt_by_image = {}
    image_id_to_key = {}
    for image_info in coco.get("images", []):
        image_path = images_dir / image_info["file_name"]
        key = image_info["file_name"]
        gt_by_image[key] = {
            "key": key,
            "path": image_path,
            "width": int(image_info["width"]),
            "height": int(image_info["height"]),
            "boxes": [],
            "labels": [],
        }
        image_id_to_key[image_info["id"]] = key

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        key = image_id_to_key.get(ann["image_id"])
        if key is None:
            continue
        if ann["category_id"] not in cat_id_to_master:
            continue
        gt_by_image[key]["boxes"].append(xywh_to_xyxy(ann["bbox"]))
        gt_by_image[key]["labels"].append(cat_id_to_master[ann["category_id"]])

    for record in gt_by_image.values():
        if record["boxes"]:
            record["boxes"] = clip_boxes_xyxy(record["boxes"], record["width"], record["height"])
            record["labels"] = np.asarray(record["labels"], dtype=np.int64)
        else:
            empty = make_empty_annotations()
            record["boxes"] = empty["boxes"]
            record["labels"] = empty["labels"]

    return dict(sorted(gt_by_image.items(), key=lambda x: x[0]))


def load_ground_truth_from_yolo(images_dir, labels_dir, yolo_class_names, master_class_names):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    if not image_paths:
        raise FileNotFoundError(f"在目录中没有找到测试图片: {images_dir}")

    master_name_to_idx = build_name_to_index(master_class_names)
    yolo_id_to_master = {}
    for idx, name in enumerate(yolo_class_names):
        key = normalize_name(name)
        if key not in master_name_to_idx:
            raise KeyError(f"YOLO GT 类别未在 master_class_names 中找到: {name}")
        yolo_id_to_master[idx] = master_name_to_idx[key]

    gt_by_image = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = []
        labels = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    xywh = [float(v) for v in parts[1:5]]
                    boxes.append(yolo_to_xyxy(xywh, width, height))
                    labels.append(yolo_id_to_master[cls_id])

        gt_by_image[image_path.name] = {
            "key": image_path.name,
            "path": image_path,
            "width": width,
            "height": height,
            "boxes": clip_boxes_xyxy(boxes, width, height) if boxes else make_empty_annotations()["boxes"],
            "labels": np.asarray(labels, dtype=np.int64) if labels else make_empty_annotations()["labels"],
        }

    return gt_by_image


def get_fasterrcnn_builder(arch_name):
    builders = {
        "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
        "fasterrcnn_mobilenet_v3_large_320_fpn": fasterrcnn_mobilenet_v3_large_320_fpn,
    }
    return builders.get(arch_name)


def build_fasterrcnn_manual(info):
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        trainable_layers=info.get("trainable_backbone_layers", 5),
    )
    anchor_sizes_per_level = info.get("anchor_sizes_per_level", ((8,), (16,), (32,), (64,), (128,)))
    anchor_ratios = info.get("anchor_ratios", (0.5, 1.0, 2.0))
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes_per_level,
        aspect_ratios=(anchor_ratios,) * len(anchor_sizes_per_level),
    )
    rpn_head = RPNHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        conv_depth=2,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=info["num_classes"],
        rpn_anchor_generator=anchor_generator,
        rpn_head=rpn_head,
        min_size=info.get("min_size", 960),
        max_size=info.get("max_size", 1536),
        rpn_pre_nms_top_n_train=info.get("rpn_pre_nms_top_n_train", 4000),
        rpn_pre_nms_top_n_test=info.get("rpn_pre_nms_top_n_test", 2000),
        rpn_post_nms_top_n_train=info.get("rpn_post_nms_top_n_train", 2000),
        rpn_post_nms_top_n_test=info.get("rpn_post_nms_top_n_test", 1000),
        rpn_fg_iou_thresh=info.get("rpn_fg_iou_thresh", 0.7),
        rpn_bg_iou_thresh=info.get("rpn_bg_iou_thresh", 0.3),
        box_score_thresh=0.0,
        box_nms_thresh=info.get("box_nms_thresh", 0.5),
        box_detections_per_img=info.get("box_detections_per_img", 500),
    )


def build_fasterrcnn_model(info):
    if info.get("arch") == "manual_resnet50_fpn":
        return build_fasterrcnn_manual(info)

    builder = get_fasterrcnn_builder(info.get("arch", "fasterrcnn_resnet50_fpn"))
    if builder is None:
        raise ValueError(f"未知 Faster R-CNN 架构: {info.get('arch')}")

    model = builder(
        weights=None,
        weights_backbone=None,
        min_size=info.get("min_size", 800),
        max_size=info.get("max_size", 1333),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, info["num_classes"])
    return model


def load_fasterrcnn_state(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return model.load_state_dict(state, strict=False)


def load_fasterrcnn_checkpoint_state(model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def infer_num_classes_from_state_dict(state):
    cls_weight = state.get("roi_heads.box_predictor.cls_score.weight")
    if cls_weight is not None and hasattr(cls_weight, "shape") and len(cls_weight.shape) == 2:
        return int(cls_weight.shape[0])

    cls_bias = state.get("roi_heads.box_predictor.cls_score.bias")
    if cls_bias is not None and hasattr(cls_bias, "shape") and len(cls_bias.shape) == 1:
        return int(cls_bias.shape[0])

    bbox_weight = state.get("roi_heads.box_predictor.bbox_pred.weight")
    if bbox_weight is not None and hasattr(bbox_weight, "shape") and len(bbox_weight.shape) == 2:
        return int(bbox_weight.shape[0] // 4)

    bbox_bias = state.get("roi_heads.box_predictor.bbox_pred.bias")
    if bbox_bias is not None and hasattr(bbox_bias, "shape") and len(bbox_bias.shape) == 1:
        return int(bbox_bias.shape[0] // 4)

    return None


def align_fasterrcnn_index_to_name(index_to_name, inferred_num_classes):
    expected_len = int(inferred_num_classes)
    current = list(index_to_name)
    if len(current) >= expected_len:
        return current[:expected_len]

    known_normalized = {normalize_name(name) for name in current}
    candidates = ["Actinomycetes"]
    for candidate in candidates:
        if len(current) >= expected_len:
            break
        if normalize_name(candidate) not in known_normalized:
            current.append(candidate)
            known_normalized.add(normalize_name(candidate))

    while len(current) < expected_len:
        current.append(f"__ignored_extra_class_{len(current)}__")

    return current


def load_model(spec, cfg):
    model_type = spec["type"]
    weights_path = Path(spec["weights"])
    device = cfg["device"]
    master_class_names = cfg["master_class_names"]

    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights_path}")

    if model_type == "yolo":
        model = YOLO(str(weights_path))
        raw_names = model.names
        index_to_name = [raw_names[i] for i in sorted(raw_names)] if isinstance(raw_names, dict) else list(raw_names)
        missing, extra = summarize_missing_classes(index_to_name, master_class_names)
        return {
            "name": spec["name"],
            "type": "yolo",
            "model": model,
            "index_to_name": index_to_name,
            "missing_classes": missing,
            "extra_classes": extra,
        }

    if model_type == "fasterrcnn":
        state = load_fasterrcnn_checkpoint_state(weights_path, device)
        inferred_num_classes = infer_num_classes_from_state_dict(state)
        build_spec = dict(spec)
        if inferred_num_classes is not None:
            build_spec["num_classes"] = inferred_num_classes

        model = build_fasterrcnn_model(build_spec)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        model.roi_heads.detections_per_img = cfg["max_det"]
        index_to_name = align_fasterrcnn_index_to_name(
            build_spec["class_names_with_background"],
            build_spec["num_classes"],
        )
        missing, extra = summarize_missing_classes(index_to_name, master_class_names)
        if missing_keys or unexpected_keys:
            print(f"[{spec['name']}] state_dict 非严格匹配: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        return {
            "name": spec["name"],
            "type": "fasterrcnn",
            "model": model,
            "index_to_name": index_to_name,
            "missing_classes": missing,
            "extra_classes": extra,
        }

    raise ValueError(f"未知模型类型: {model_type}")


def convert_labels_to_master(raw_labels, index_to_name, master_class_names):
    master_name_to_idx = build_name_to_index(master_class_names)
    converted = []
    keep_mask = []
    dropped = []
    background_alias = {"__background__", "background", "bg"}

    for raw_label in raw_labels:
        raw_label = int(raw_label)
        if raw_label < 0 or raw_label >= len(index_to_name):
            keep_mask.append(False)
            dropped.append(f"id={raw_label}")
            continue
        class_name = index_to_name[raw_label]
        normalized = normalize_name(class_name)
        if normalized in background_alias:
            keep_mask.append(False)
            continue
        if normalized not in master_name_to_idx:
            keep_mask.append(False)
            dropped.append(class_name)
            continue
        keep_mask.append(True)
        converted.append(master_name_to_idx[normalized])

    return np.asarray(converted, dtype=np.int64), np.asarray(keep_mask, dtype=bool), sorted(set(dropped))


def run_inference(model_pack, image_path, cfg):
    if model_pack["type"] == "yolo":
        result = model_pack["model"].predict(
            source=str(image_path),
            conf=cfg["raw_pred_conf"],
            iou=cfg["nms_iou"],
            max_det=cfg["max_det"],
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "scores": np.zeros((0,), dtype=np.float32),
                "dropped_labels": [],
            }
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        raw_labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        labels, keep_mask, dropped = convert_labels_to_master(raw_labels, model_pack["index_to_name"], cfg["master_class_names"])
        return {
            "boxes": boxes[keep_mask],
            "labels": labels,
            "scores": scores[keep_mask],
            "dropped_labels": dropped,
        }

    device = cfg["device"]
    with Image.open(image_path).convert("RGB") as image:
        tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model_pack["model"]([tensor])[0]
    boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
    raw_labels = output["labels"].detach().cpu().numpy().astype(np.int64)
    scores = output["scores"].detach().cpu().numpy().astype(np.float32)
    labels, keep_mask, dropped = convert_labels_to_master(raw_labels, model_pack["index_to_name"], cfg["master_class_names"])
    return {
        "boxes": boxes[keep_mask],
        "labels": labels,
        "scores": scores[keep_mask],
        "dropped_labels": dropped,
    }


def infer_dataset(model_pack, gt_by_image, cfg):
    preds_by_image = {}
    dropped_names = set()
    iterator = tqdm(gt_by_image.items(), total=len(gt_by_image), desc=f"Infer {model_pack['name']}")
    for key, gt in iterator:
        pred = run_inference(model_pack, gt["path"], cfg)
        preds_by_image[key] = {
            "boxes": clip_boxes_xyxy(pred["boxes"], gt["width"], gt["height"]),
            "labels": pred["labels"],
            "scores": pred["scores"],
        }
        dropped_names.update(pred.get("dropped_labels", []))
    return preds_by_image, sorted(dropped_names)


def load_ground_truth(cfg):
    if str(cfg["gt_source"]).strip().lower() == "coco":
        return load_ground_truth_from_coco(
            cfg["test_images_dir"],
            cfg["coco_ann_path"],
            cfg["master_class_names"],
            strict_category_check=cfg.get("strict_gt_category_check", False),
        )
    return load_ground_truth_from_yolo(
        cfg["test_images_dir"],
        cfg["yolo_labels_dir"],
        cfg["yolo_gt_class_names"],
        cfg["master_class_names"],
    )


def build_detection_confusion_matrix(gt_by_image, preds_by_image, class_names, conf_threshold=0.25, iou_threshold=0.5):
    num_classes = len(class_names)
    bg_idx = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for key, gt in gt_by_image.items():
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred = filter_predictions(preds_by_image[key], conf_threshold)
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            for pred_label in pred_labels:
                matrix[bg_idx, pred_label] += 1
            continue
        if len(pred_boxes) == 0:
            for gt_label in gt_labels:
                matrix[gt_label, bg_idx] += 1
            continue

        ious = box_iou(gt_boxes, pred_boxes)
        candidates = []
        for gt_idx in range(len(gt_boxes)):
            for pred_idx in range(len(pred_boxes)):
                iou = float(ious[gt_idx, pred_idx])
                if iou >= iou_threshold:
                    candidates.append((iou, gt_idx, pred_idx))
        candidates.sort(key=lambda x: x[0], reverse=True)

        used_gt = set()
        used_pred = set()
        for _, gt_idx, pred_idx in candidates:
            if gt_idx in used_gt or pred_idx in used_pred:
                continue
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
            matrix[gt_labels[gt_idx], pred_labels[pred_idx]] += 1

        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in used_gt:
                matrix[gt_label, bg_idx] += 1
        for pred_idx, pred_label in enumerate(pred_labels):
            if pred_idx not in used_pred:
                matrix[bg_idx, pred_label] += 1

    return matrix


def compute_pr_for_class(gt_by_image, preds_by_image, class_idx, iou_threshold=0.5):
    gt_boxes_per_image = {}
    positives = 0
    for key, gt in gt_by_image.items():
        mask = gt["labels"] == class_idx
        boxes = gt["boxes"][mask]
        gt_boxes_per_image[key] = boxes
        positives += len(boxes)

    detections = []
    for key, pred in preds_by_image.items():
        mask = pred["labels"] == class_idx
        boxes = pred["boxes"][mask]
        scores = pred["scores"][mask]
        for box, score in zip(boxes, scores):
            detections.append((float(score), key, box.astype(np.float32)))

    detections.sort(key=lambda x: x[0], reverse=True)

    if positives == 0:
        return {
            "recall": np.array([], dtype=np.float32),
            "precision": np.array([], dtype=np.float32),
            "scores": np.array([], dtype=np.float32),
            "ap": np.nan,
            "num_gt": 0,
        }

    matched = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_boxes_per_image.items()}
    tp = np.zeros(len(detections), dtype=np.float32)
    fp = np.zeros(len(detections), dtype=np.float32)
    scores = np.zeros(len(detections), dtype=np.float32)

    for idx, (score, key, pred_box) in enumerate(detections):
        scores[idx] = score
        gt_boxes = gt_boxes_per_image[key]
        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue
        ious = box_iou(np.asarray([pred_box], dtype=np.float32), gt_boxes)[0]
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])
        if best_iou >= iou_threshold and not matched[key][best_gt]:
            matched[key][best_gt] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(float(positives), 1e-9)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return {
        "recall": recall,
        "precision": precision,
        "scores": scores,
        "ap": compute_ap(recall, precision),
        "num_gt": positives,
    }


def metrics_from_confusion_matrix(matrix, class_names):
    num_classes = len(class_names)
    tp = np.diag(matrix[:num_classes, :num_classes]).astype(np.float64)
    fp = matrix[:, :num_classes].sum(axis=0).astype(np.float64) - tp
    fn = matrix[:num_classes, :].sum(axis=1).astype(np.float64) - tp
    support = matrix[:num_classes, :].sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    per_class = pd.DataFrame(
        {
            "class_name": class_names,
            "support": support.astype(int),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    tp_sum = float(tp.sum())
    fp_sum = float(fp.sum())
    fn_sum = float(fn.sum())
    micro_precision = safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    valid_mask = support > 0
    macro_precision = float(precision[valid_mask].mean()) if valid_mask.any() else 0.0
    macro_recall = float(recall[valid_mask].mean()) if valid_mask.any() else 0.0
    macro_f1 = float(f1[valid_mask].mean()) if valid_mask.any() else 0.0

    return {
        "per_class": per_class,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_count_metrics(gt_by_image, preds_by_image, class_names, conf_threshold=0.25):
    rows = []
    num_classes = len(class_names)
    gt_totals = np.zeros(num_classes, dtype=np.int64)
    pred_totals = np.zeros(num_classes, dtype=np.int64)

    for key, gt in gt_by_image.items():
        pred = filter_predictions(preds_by_image[key], conf_threshold)
        gt_counts = np.bincount(gt["labels"], minlength=num_classes)
        pred_counts = np.bincount(pred["labels"], minlength=num_classes)
        gt_totals += gt_counts
        pred_totals += pred_counts
        rows.append(
            {
                "image_name": key,
                "gt_total": int(gt_counts.sum()),
                "pred_total": int(pred_counts.sum()),
                "abs_error": int(abs(int(pred_counts.sum()) - int(gt_counts.sum()))),
            }
        )

    count_df = pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)
    total_error = count_df["pred_total"] - count_df["gt_total"]
    mae = float(np.abs(total_error).mean()) if len(count_df) else 0.0
    rmse = float(np.sqrt(np.mean(np.square(total_error)))) if len(count_df) else 0.0
    bias = float(total_error.mean()) if len(count_df) else 0.0

    per_class_count = pd.DataFrame(
        {
            "class_name": class_names,
            "gt_count": gt_totals,
            "pred_count": pred_totals,
            "abs_error": np.abs(pred_totals - gt_totals),
        }
    )

    return {
        "per_image": count_df,
        "per_class": per_class_count,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def threshold_sweep_metrics(gt_by_image, preds_by_image, class_names, thresholds, iou_threshold=0.5):
    rows = []
    for threshold in thresholds:
        matrix = build_detection_confusion_matrix(
            gt_by_image,
            preds_by_image,
            class_names,
            conf_threshold=float(threshold),
            iou_threshold=iou_threshold,
        )
        base_metrics = metrics_from_confusion_matrix(matrix, class_names)
        rows.append(
            {
                "threshold": float(threshold),
                "micro_precision": base_metrics["micro_precision"],
                "micro_recall": base_metrics["micro_recall"],
                "micro_f1": base_metrics["micro_f1"],
                "macro_precision": base_metrics["macro_precision"],
                "macro_recall": base_metrics["macro_recall"],
                "macro_f1": base_metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(model_name, gt_by_image, preds_by_image, cfg):
    class_names = cfg["master_class_names"]
    ap50_list = []
    ap5095_list = []
    pr_curves = {}
    supports = []

    for class_idx, class_name in enumerate(class_names):
        curves = [compute_pr_for_class(gt_by_image, preds_by_image, class_idx, float(iou_thr)) for iou_thr in cfg["map_iou_thresholds"]]
        ap50 = curves[0]["ap"]
        ap5095 = float(np.nanmean([curve["ap"] for curve in curves])) if curves[0]["num_gt"] > 0 else np.nan
        ap50_list.append(ap50)
        ap5095_list.append(ap5095)
        pr_curves[class_name] = curves[0]
        supports.append(curves[0]["num_gt"])

    confusion = build_detection_confusion_matrix(
        gt_by_image,
        preds_by_image,
        class_names,
        conf_threshold=cfg["confusion_conf"],
        iou_threshold=cfg["confusion_iou"],
    )
    base_metrics = metrics_from_confusion_matrix(confusion, class_names)
    count_metrics = compute_count_metrics(gt_by_image, preds_by_image, class_names, conf_threshold=cfg["count_conf"])
    sweep_df = threshold_sweep_metrics(gt_by_image, preds_by_image, class_names, cfg["threshold_sweep"], iou_threshold=cfg["confusion_iou"])

    per_class_df = base_metrics["per_class"].copy()
    per_class_df["ap50"] = ap50_list
    per_class_df["ap50_95"] = ap5095_list
    per_class_df["gt_instances"] = supports

    valid_ap50 = per_class_df.loc[per_class_df["gt_instances"] > 0, "ap50"]
    valid_ap5095 = per_class_df.loc[per_class_df["gt_instances"] > 0, "ap50_95"]

    return {
        "model_name": model_name,
        "confusion_matrix": confusion,
        "metrics": base_metrics,
        "per_class": per_class_df,
        "pr_curves": pr_curves,
        "mAP50": float(valid_ap50.mean()) if len(valid_ap50) else np.nan,
        "mAP50_95": float(valid_ap5095.mean()) if len(valid_ap5095) else np.nan,
        "count_metrics": count_metrics,
        "threshold_sweep": sweep_df,
    }


def plot_confusion_matrix(matrix, class_names, title, normalize=False, save_path=None):
    labels = class_names + ["background"]
    matrix = matrix.astype(np.float64)
    display_matrix = matrix.copy()
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(display_matrix, row_sums, out=np.zeros_like(display_matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(display_matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fmt = ".2f" if normalize else ".0f"
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            value = display_matrix[i, j]
            text = format(value, fmt) if normalize else str(int(round(value)))
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_pr_curves(result, cfg, max_classes=10, save_path=None):
    per_class = result["per_class"].sort_values(["gt_instances", "ap50"], ascending=[False, False])
    plot_classes = per_class.loc[per_class["gt_instances"] > 0, "class_name"].tolist()[:max_classes]
    fig, ax = plt.subplots(figsize=(10, 8))
    for class_name in plot_classes:
        curve = result["pr_curves"][class_name]
        if len(curve["recall"]) == 0:
            continue
        ax.plot(curve["recall"], curve["precision"], linewidth=2, label=f"{class_name} (AP50={curve['ap']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves @ IoU={cfg['pr_curve_iou']:.2f} | {result['model_name']}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_ap_bar(result, metric_column="ap50", save_path=None):
    df = result["per_class"].copy()
    df = df[df["gt_instances"] > 0].sort_values(metric_column, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["class_name"], df[metric_column], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_column)
    ax.set_title(f"Per-class {metric_column} | {result['model_name']}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_threshold_sweep(result, save_path=None):
    df = result["threshold_sweep"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold"], df["micro_precision"], label="Micro Precision", linewidth=2)
    ax.plot(df["threshold"], df["micro_recall"], label="Micro Recall", linewidth=2)
    ax.plot(df["threshold"], df["micro_f1"], label="Micro F1", linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold Sweep | {result['model_name']}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_count_scatter(result, save_path=None):
    df = result["count_metrics"]["per_image"]
    max_value = max(df["gt_total"].max() if len(df) else 0, df["pred_total"].max() if len(df) else 0, 1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["gt_total"], df["pred_total"], alpha=0.8, s=40, color="#F58518")
    ax.plot([0, max_value], [0, max_value], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("GT Count / Image")
    ax.set_ylabel("Pred Count / Image")
    ax.set_title(
        f"Counting Scatter | {result['model_name']}\n"
        f"MAE={result['count_metrics']['mae']:.2f}, "
        f"RMSE={result['count_metrics']['rmse']:.2f}, "
        f"Bias={result['count_metrics']['bias']:.2f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_gt_vs_pred_class_counts(result, save_path=None):
    df = result["count_metrics"]["per_class"].copy()
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df["gt_count"], width=width, label="GT", color="#4C78A8")
    ax.bar(x + width / 2, df["pred_count"], width=width, label="Pred", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"GT vs Pred Class Counts | {result['model_name']}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_result_artifacts(result, cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = cfg["master_class_names"]

    result["per_class"].to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    result["threshold_sweep"].to_csv(output_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    result["count_metrics"]["per_image"].to_csv(output_dir / "count_per_image.csv", index=False, encoding="utf-8-sig")
    result["count_metrics"]["per_class"].to_csv(output_dir / "count_per_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        result["confusion_matrix"],
        index=class_names + ["background"],
        columns=class_names + ["background"],
    ).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    plot_confusion_matrix(result["confusion_matrix"], class_names, f"Confusion Matrix | {result['model_name']}", False, output_dir / "confusion_matrix.png")
    plot_confusion_matrix(result["confusion_matrix"], class_names, f"Normalized Confusion Matrix | {result['model_name']}", True, output_dir / "confusion_matrix_normalized.png")
    plot_pr_curves(result, cfg, save_path=output_dir / "pr_curves.png")
    plot_ap_bar(result, "ap50", output_dir / "ap50_bar.png")
    plot_ap_bar(result, "ap50_95", output_dir / "ap50_95_bar.png")
    plot_threshold_sweep(result, output_dir / "threshold_sweep.png")
    plot_count_scatter(result, output_dir / "count_scatter.png")
    plot_gt_vs_pred_class_counts(result, output_dir / "class_count_bar.png")


def plot_model_comparison(summary_df, save_path=None):
    df = summary_df.sort_values("mAP50", ascending=False)
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, df["mAP50"], width=width, label="mAP@0.5", color="#4C78A8")
    ax.bar(x + width / 2, df["mAP50_95"], width=width, label="mAP@0.5:0.95", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def run_full_evaluation(cfg):
    image_dir, enabled_models = validate_config(cfg)
    gt_by_image = load_ground_truth(cfg)
    print(f"Loaded {len(gt_by_image)} test images from: {image_dir}")

    missing_images = [key for key, record in gt_by_image.items() if not Path(record["path"]).exists()]
    if missing_images:
        raise FileNotFoundError(f"以下图片在测试集目录中不存在，例如: {missing_images[:5]}")

    summary_rows = []
    all_results = {}

    for spec in enabled_models:
        print("=" * 100)
        print(f"Evaluating: {spec['name']}")
        model_pack = load_model(spec, cfg)
        if model_pack["missing_classes"]:
            print(f"[{spec['name']}] 模型中缺失的主类别: {model_pack['missing_classes']}")
        if model_pack["extra_classes"]:
            print(f"[{spec['name']}] 模型中存在但不在主类别列表中的类别: {model_pack['extra_classes']}")

        preds_by_image, dropped_names = infer_dataset(model_pack, gt_by_image, cfg)
        if dropped_names:
            print(f"[{spec['name']}] 以下预测类别被忽略，因为不在主类别列表中: {dropped_names}")

        result = evaluate_predictions(spec["name"], gt_by_image, preds_by_image, cfg)
        save_dir = Path(cfg["output_dir"]) / spec["name"]
        save_result_artifacts(result, cfg, save_dir)
        all_results[spec["name"]] = result

        summary_rows.append(
            {
                "model_name": spec["name"],
                "mAP50": result["mAP50"],
                "mAP50_95": result["mAP50_95"],
                "micro_precision": result["metrics"]["micro_precision"],
                "micro_recall": result["metrics"]["micro_recall"],
                "micro_f1": result["metrics"]["micro_f1"],
                "macro_precision": result["metrics"]["macro_precision"],
                "macro_recall": result["metrics"]["macro_recall"],
                "macro_f1": result["metrics"]["macro_f1"],
                "count_mae": result["count_metrics"]["mae"],
                "count_rmse": result["count_metrics"]["rmse"],
                "count_bias": result["count_metrics"]["bias"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["mAP50", "micro_f1"], ascending=False).reset_index(drop=True)
    summary_df.to_csv(Path(cfg["output_dir"]) / "model_summary.csv", index=False, encoding="utf-8-sig")
    plot_model_comparison(summary_df, Path(cfg["output_dir"]) / "model_comparison.png")
    return summary_df, all_results
