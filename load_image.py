import cv2

def load_image_rgb(path: str)->np.ndarray:
  bgr=cv2.imread(path,cv2.IMREAD_COLOR)
  if bgr is None:
    raise FileNotFoundError(f"Image not found :{path}")
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  return rgb
  
