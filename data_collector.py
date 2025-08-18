import cv2
import os
import numpy as np
import time
from keypoint_feature_extractor import HandFeatureExtractor

class GestureDataCollector:
    def __init__(self, output_dir="gesture_data"):
        """初始化手势数据收集器"""
        self.output_dir = output_dir
        self.feature_extractor = HandFeatureExtractor()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.gestures = {
            0: "拳头",
            1: "手掌",
            2: "指向",
            3: "OK",
            4: "胜利",
            5: "大拇指向上"
        }
    
    def collect_data(self, gesture_id, num_samples=100):
        """收集特定手势的数据

        参数:
            gesture_id: 手势ID
            num_samples: 要收集的样本数
        """
        # 中文字体路径（可根据实际情况调整）
        font_path = r"C:\Windows\Fonts\simhei.ttf"
        font_size = 32
        def cv2_add_chinese_text(img, text, position, color=(255,0,0), font_size=32):
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(font_path, font_size)
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if gesture_id not in self.gestures:
            print(f"未知手势ID: {gesture_id}")
            return
        
        gesture_name = self.gestures[gesture_id]
        print(f"准备收集 '{gesture_name}' 手势的数据。请将手放在摄像头前。")
        print(f"按'空格'开始，按'q'退出。")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        print("摄像头打开状态：", cap.isOpened())
        
        # 等待用户准备好
        collecting = False
        collected = 0
        
        features_list = []
        
        from PIL import ImageFont, ImageDraw, Image
        while True:
            ret, frame = cap.read()
            print("帧读取状态：", ret)
            if not ret:
                break
            
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            
            # 提取特征
            features, hand_present = self.feature_extractor.extract_keypoint_features(frame)
            
            # 显示状态
            if collecting:
                status_text = f"收集中: {collected}/{num_samples}"
                color = (0, 255, 0)
            else:
                status_text = "按空格开始收集"
                color = (0, 0, 255)
            
            frame = cv2_add_chinese_text(frame, status_text, (10, 30), color=color, font_size=font_size)
            frame = cv2_add_chinese_text(frame, f"手势: {gesture_name}", (10, 70), color=color, font_size=font_size)
            
            # 显示是否检测到手
            if hand_present:
                frame = cv2_add_chinese_text(frame, "手部检测: 是", (10, 110), color=(0,255,0), font_size=font_size)
            else:
                frame = cv2_add_chinese_text(frame, "手部检测: 否", (10, 110), color=(0,0,255), font_size=font_size)
            
            # 显示图像（窗口名称也用中文）
            cv2.imshow("手势数据收集", frame)
            
            # 收集数据
            if collecting and hand_present and features.size > 0:
                features_list.append(features)
                collected += 1
                
                # 提供反馈
                print(f"\r已收集: {collected}/{num_samples}", end="")
                
                # 检查是否收集完成
                if collected >= num_samples:
                    collecting = False
                    print(f"\n已完成 '{gesture_name}' 的数据收集!")
                    break
                
                # 稍作等待，避免收集太快
                time.sleep(0.1)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键
                collecting = True
                print(f"开始收集 '{gesture_name}' 数据...")
            elif key == ord('q'):  # 'q'键
                break
        
        # 保存收集到的数据
        if features_list:
            expected_len = len(features_list[0])
            filtered_features = [f for f in features_list if len(f) == expected_len and f is not None]
            if len(filtered_features) < len(features_list):
                print(f"有 {len(features_list) - len(filtered_features)} 个特征长度异常或无效，已自动过滤。")
            if filtered_features:
                features_array = np.array(filtered_features)
                print("最终特征 shape:", features_array.shape)
                file_path = os.path.abspath(os.path.join(self.output_dir, f"gesture_{gesture_id}_{gesture_name}.npz"))
                print(f"数据将保存到: {file_path}")
                np.savez(file_path, gesture_id=gesture_id, gesture_name=gesture_name, features=features_array)
                print(f"数据已保存到: {file_path}")
            else:
                print("没有有效特征可保存，请重新采集。")
        # 询问用户是否继续
        response = input("继续收集下一个手势? (y/n): ")
        if response.lower() != 'y':
            return

if __name__ == "__main__":
    collector = GestureDataCollector()
    for gesture_id in collector.gestures:
        collector.collect_data(gesture_id, num_samples=100)
        response = input("继续收集下一个手势? (y/n): ")
        if response.lower() != 'y':
            break