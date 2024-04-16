def get_bbox_center(bbox):
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return x_center, y_center

def get_z(bbox):
  z = depth_map.get_value(get_bbox_center(bbox))
  return z

