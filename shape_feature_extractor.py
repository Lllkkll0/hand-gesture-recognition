def extract_shape_features(self, image, hand_landmarks):
    """提取手部形状特征"""
    h, w, c = image.shape
    
    # 创建手部掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 提取关键点坐标
    points = []
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        points.append((x, y))
    
    # 创建凸包
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    
    # 在掩码上填充手部区域
    cv2.fillConvexPoly(mask, hull, 255)
    
    # 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    features = []
    if contours:
        contour = contours[0]
        
        # 1. 面积和周长
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 2. 圆形度 (4π·Area/Perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 3. 矩形度 (Area/面积最小矩形的面积)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 4. 长宽比
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 5. 凸度 (轮廓面积/凸包面积)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 6. Hu矩（形状不变矩）
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        
        # 将所有特征添加到列表中
        features.extend([area, perimeter, circularity, rectangularity, 
                         aspect_ratio, convexity])
        features.extend(hu_moments.flatten())
    
    return np.array(features)