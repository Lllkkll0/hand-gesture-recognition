import cv2
import mediapipe as mp
import numpy as np
import argparse

class VideoHandContourDetector:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化视频手部轮廓检测器
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_video(self, video_path, output_path=None):
        """
        处理视频文件
        
        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化视频写入器（如果需要）
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理每一帧
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # 处理图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # 创建遮罩
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 如果检测到手部
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制关键点
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 提取关键点坐标
                    points = []
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        points.append((x, y))
                    
                    # 连接关键点创建遮罩
                    points = np.array(points, dtype=np.int32)
                    
                    # 使用凸包算法获取手部轮廓
                    hull = cv2.convexHull(points)
                    
                    # 在遮罩上绘制手部区域
                    cv2.fillConvexPoly(mask, hull, 255)
                    
                    # 找到轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 绘制轮廓
                    if contours:
                        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # 显示进度
            frame_idx += 1
            print(f"\r处理中: {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)", end="")
            
            # 显示结果
            cv2.imshow("Hand Contour Detection", frame)
            
            # 写入输出视频（如果需要）
            if writer:
                writer.write(frame)
                
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n处理完成")
        
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从视频中检测手部轮廓")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, help="输出视频路径")
    args = parser.parse_args()
    
    # 初始化检测器
    detector = VideoHandContourDetector()
    
    # 处理视频
    output_path = args.output if args.output else "G:\\hand\\hand video resluts\\output.mp4"
    detector.process_video(args.input, output_path)

if __name__ == "__main__":
    main()