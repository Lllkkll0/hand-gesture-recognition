import cv2
import mediapipe as mp
import numpy as np

class ImprovedHandContourDetector:
    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化改进版手部轮廓检测器
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def process_image(self, img):
        """
        处理图像并返回结果
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(img_rgb)
    
    def draw_contours(self, img, results, draw_landmarks=True, draw_contour=True):
        """
        绘制手部轮廓
        
        参数:
            img: 输入图像
            results: MediaPipe处理结果
            draw_landmarks: 是否绘制关键点
            draw_contour: 是否绘制轮廓
            
        返回:
            处理后的图像
        """
        # 创建遮罩
        h, w, c = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制关键点
                if draw_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 提取关键点坐标
                points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
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
                if draw_contour and contours:
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                    
                    # 可选：绘制凸包
                    # cv2.drawContours(img, [hull], 0, (255, 0, 0), 2)
        
        return img

def main():
    # 选择视频来源 (0 表示默认摄像头)
    cap = cv2.VideoCapture(0)
    
    # 初始化检测器
    detector = ImprovedHandContourDetector()
    
    while True:
        success, img = cap.read()
        if not success:
            print("无法获取图像")
            break
            
        # 处理图像
        results = detector.process_image(img)
        
        # 绘制手部轮廓
        img = detector.draw_contours(img, results)
        
        # 显示FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # 显示结果
        cv2.imshow("Improved Hand Contour Detection", img)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()