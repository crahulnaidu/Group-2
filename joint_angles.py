import numpy as np

#function to convert 25 key points to 13 joint angles

def convert_to_joint_angles(skeleton:np.ndarray,conf_thresh:float=0.1)->np.ndarray:
  #checking if the skeleton is empty 
  if skeleton is None or not isinstance(skeleton,np.ndarray) or skeleton.size==0:
    return np.array([])

  #bool variable to check if the no of columns is >=3
  has_conf=skeleton.shape[1]>=3 

  #visible function is used to check if a key point has probability or confidence greater than threshold or is a valid keypoint in 
  the np array
  def visible(i):
    if i<0 or i>=skeleton.shape[0]:
      return False
    if not has_conf:
      return not(skeleton[i,0]==0 and skeleton[i,1]==0)
      
    return skeleton[i,2]>=conf_thresh

  
  def angle_btw(p1_id,p2_id,p3_id):
    if not(visible(p1_id) and visible(p2_id) and visible(p3_id)):
      return 0.0

    #getting the corresponding keypoint array from the np array skeleton
    
    p1=skeleton[p1_id,:2].astype(np.float64)
    p2=skeleton[p2_id,:2].astype(np.float64)
    p3=skeleton[p3_id,:2].astype(np.float64)

    #calculating the difference vector
    v1=p1-p2
    v2=p3-p2
    
    n1=np.linalg.norm(v1)
    n2=np.linalg.norm(v2)

    #if either of the vector length is smaller than 10^-6 then simply return 0.0
    
    if n1<1e-6 or n2<1e-6:
      return 0.0

    #calculating the angle between the vector's using their tangent
    
    a1=np.arctan2(v1[1],v1[0])
    a2=np.arctan2(v2[1],v2[0])
    
    ang=a2-a1
    ang=(ang+np.pi)%(2*np.pi)-np.pi
    
    return ang

  def idx(i):
    return i

  #calculating the angles for 13 joints and storing them in the angle dictionary 
  
  angles={}  
  angles['neck'] = angle_btw(idx(0), idx(1), idx(8))
  
  angles['right_shoulder'] = angle_btw(idx(1), idx(2), idx(3))
  angles['left_shoulder'] = angle_btw(idx(1), idx(5), idx(6))
  
  angles['right_elbow'] = angle_btw(idx(2), idx(3), idx(4))
  angles['left_elbow'] = angle_btw(idx(5), idx(6), idx(7))
  
  angles['midhip'] = angle_btw(idx(1), idx(8), idx(9)) 
  
  angles['right_hip'] = angle_btw(idx(8), idx(9), idx(10))
  angles['left_hip'] = angle_btw(idx(8), idx(12), idx(13))
  
  angles['right_knee'] = angle_btw(idx(9), idx(10), idx(11))
  angles['left_knee'] = angle_btw(idx(12), idx(13), idx(14))
  
  angles['right_ankle'] = angle_btw(idx(10), idx(11), idx(24))
  angles['left_ankle'] = angle_btw(idx(13), idx(14), idx(21))

  angles['right_wrist'] = angle_btw(idx(3), idx(4), idx(22)) 
  angles['left_wrist'] = angle_btw(idx(6), idx(7), idx(19))
  
  initial_pose_vector = np.array([
        angles.get('neck', 0.0),
        angles.get('right_shoulder', 0.0),
        angles.get('left_shoulder', 0.0),
        angles.get('right_elbow', 0.0),
        angles.get('left_elbow', 0.0),
        angles.get('midhip', 0.0),
        angles.get('right_hip', 0.0),
        angles.get('left_hip', 0.0),
        angles.get('right_knee', 0.0),
        angles.get('left_knee', 0.0),
        angles.get('right_ankle', 0.0),
        angles.get('left_ankle', 0.0),
        angles.get('right_wrist', 0.0),
        angles.get('left_wrist', 0.0),
    ], dtype=np.float64)

  return initial_pose_vector
  
      
    
    


    


      


