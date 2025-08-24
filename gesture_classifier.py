import numpy as np
import cv2
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keypoint_feature_extractor import HandFeatureExtractor

class GestureRecognizer:
    def __init__(self, model_path=None):
        """初始化手势识别器"""
        # 初始化特征提取器
        self.feature_extractor = HandFeatureExtractor()
        
        # 初始化分类器
        self.classifier = None
        self.scaler = StandardScaler()
        
        # 如果提供了模型路径，加载预训练模型
        if model_path:
            self.load_model(model_path)
            
        # 手势标签映射
        self.gesture_labels = {
            0: "拳头",
            1: "手掌",
            2: "指向",
            3: "OK",
            4: "胜利",
            5: "大拇指向上",
            # 可以根据需要添加更多手势
        }
    
    def preprocess_features(self, features):
        """预处理特征"""
        # 检查特征是否为空
        if features.size == 0:
            return np.array([])
        
        # 标准化特征
        return self.scaler.transform([features])[0]
    
    def train(self, X, y):
        """训练手势分类器
        
        参数:
            X: 特征矩阵，每行是一组特征
            y: 标签向量
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 初始化并训练随机森林分类器
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_scaled, y)
        
        print(f"分类器训练完成，准确率: {self.classifier.score(X_scaled, y):.4f}")
    
    def predict(self, features):
        """预测手势类别
        
        参数:
            features: 提取的特征向量
            
        返回:
            预测的手势标签
        """
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
        
        if features.size == 0:
            return None
        
        # 预处理特征
        features_scaled = self.preprocess_features(features)
        
        if features_scaled.size == 0:
            return None
        
        # 预测
        prediction = self.classifier.predict([features_scaled])[0]
        
        return prediction
    
    def recognize_gesture(self, features):
        """识别图像中的手势
        
        参数:
            image: 输入图像
            
        返回:
            识别到的手势标签和置信度
        """
        # 校验特征长度是否与 scaler 期望一致
        if features is None or len(features) != self.scaler.n_features_in_:
            return None, 0.0
        
        # 预测
        prediction = self.predict(features)
        if prediction is None:
            return None, 0.0
        proba = self.classifier.predict_proba([self.preprocess_features(features)])[0]
        if prediction >= len(proba):
            print(f"警告: 预测类别索引 {prediction} 超出概率数组长度 {len(proba)}，返回最低置信度。")
            confidence = min(proba)
        else:
            confidence = proba[prediction]
        label = self.gesture_labels[prediction] if isinstance(self.gesture_labels, (list, tuple)) else self.gesture_labels.get(prediction, "未知")
        return label, confidence
    
    def save_model(self, model_path):
        """保存模型到文件"""
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'gesture_labels': self.gesture_labels
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """从文件加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            # 强制赋值 gesture_labels，避免属性缺失
            self.gesture_labels = model_data['gesture_labels'] if 'gesture_labels' in model_data else self.gesture_labels
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {e}")