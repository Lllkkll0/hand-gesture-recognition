import cv2
from PIL import ImageFont, ImageDraw, Image
def cv2_add_chinese_text(img, text, position, color=(255,0,0), font_size=32, font_path=r"C:\Windows\Fonts\simhei.ttf"):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
import numpy as np
import os
import glob
from gesture_classifier import GestureRecognizer

def train_gesture_model():
    """训练手势识别模型"""
    # 初始化分类器
    recognizer = GestureRecognizer()
    
    # 加载所有收集的数据
    data_dir = r"G:/hand/gesture_data"
    data_files = glob.glob(os.path.join(data_dir, "gesture_*.npz"))
    
    if not data_files:
        print("没有找到训练数据，请先收集数据")
        return False
    
    # 准备训练数据
    X = []  # 特征
    y = []  # 标签
    
    for file_path in data_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            gesture_id = data['gesture_id']
            features = data['features']
            
            for feature in features:
                X.append(feature)
                y.append(gesture_id)
                
            print(f"加载了 {len(features)} 个 '{data['gesture_name']}' 手势的样本")
        except Exception as e:
            print(f"加载 {file_path} 时出错: {e}")
    
    # 检查数据是否足够
    if len(X) < 20:
        print("训练数据不足，请收集更多样本")
        return False
    
    # 过滤特征长度异常的样本
    expected_len = len(X[0])
    filtered_X = [x for x in X if len(x) == expected_len]
    filtered_y = [y[i] for i, x in enumerate(X) if len(x) == expected_len]
    if len(filtered_X) < len(X):
        print(f"有 {len(X) - len(filtered_X)} 个样本特征长度异常，已自动过滤。")
    X = np.array(filtered_X)
    y = np.array(filtered_y)
    
    # 训练模型
    recognizer.train(X, y)
    
    # 保存模型
    recognizer.save_model("gesture_model.pkl")
    
    return True

def real_time_recognition():
    import time
    gesture_log = []
    """实时手势识别"""
    # 加载模型
    recognizer = GestureRecognizer("gesture_model.pkl")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    last_gestures = dict()  # {hand_index: (gesture, start_time)}
    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # 摄像头异常时跳过本帧

        frame = cv2.flip(frame, 1)

        features_list, hand_count = recognizer.feature_extractor.extract_keypoint_features(frame)
        now = time.time()
        if hand_count == 0 or not features_list:
            frame = cv2_add_chinese_text(frame, "未检测到手势", (10, 30), color=(0,0,255), font_size=32)
            # 记录无手势
            gesture_log.append({"time": now, "hand_index": None, "gesture": None, "confidence": None})
        else:
            for idx, features in enumerate(features_list):
                gesture, confidence = recognizer.recognize_gesture(features)
                frame = cv2_add_chinese_text(frame, f"手{idx+1}: {gesture} 置信度: {confidence:.2f}", (10, 30+idx*40), color=(0,255,0), font_size=32)
                gesture_log.append({"time": now, "hand_index": idx, "gesture": gesture, "confidence": confidence})
                # 统计时长
                if idx not in last_gestures or last_gestures[idx][0] != gesture:
                    last_gestures[idx] = (gesture, now)
                else:
                    # 手势未变，持续统计
                    pass

        cv2.imshow("手势识别", frame)
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 统计每个手势的出现顺序与时长
    print("\n手势日志统计：")
    from collections import defaultdict
    import os, json
    gesture_durations = defaultdict(list)  # {gesture: [(start, end, hand_index)]}
    for idx in set([log["hand_index"] for log in gesture_log if log["hand_index"] is not None]):
        current_gesture = None
        start_time = None
        for log in gesture_log:
            if log["hand_index"] != idx:
                continue
            if log["gesture"] != current_gesture:
                if current_gesture is not None:
                    gesture_durations[current_gesture].append((start_time, log["time"], idx))
                current_gesture = log["gesture"]
                start_time = log["time"]
        # 最后一个手势
        if current_gesture is not None and start_time is not None:
            gesture_durations[current_gesture].append((start_time, gesture_log[-1]["time"], idx))
    for gesture, durations in gesture_durations.items():
        for start, end, idx in durations:
            print(f"手{idx+1}：手势 '{gesture}' 持续 {end-start:.2f} 秒")
    # 保存日志到文件
    log_dir = r"G:/hand/log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gesture_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(gesture_log, f, ensure_ascii=False, indent=2)
    print(f"手势日志已保存到: {log_path}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 询问用户是训练还是识别
    choice = input("选择操作: (1) 训练模型 (2) 实时识别: ")
    
    if choice == '1':
        print("开始训练模型...")
        if train_gesture_model():
            print("模型训练完成!")
    elif choice == '2':
        print("启动实时手势识别...")
        real_time_recognition()
    else:
        print("无效选择")