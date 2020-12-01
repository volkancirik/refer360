'''
Panoramic camera to generate perspective for equirectangular images.
adapted from here :  https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
'''
import cv2
import numpy as np


class PanoramicCamera:
  def __init__(self, fov=90, output_image_shape=(400, 400)):
    """Init the camera.
    """
    self.fov = fov
    self.output_image_h = output_image_shape[0]
    self.output_image_w = output_image_shape[1]

    self.lng_map = None
    self.lat_map = None

    self.img_path = ''
    self._img = None

  def load_img(self, img_path, gt_loc=None, convert_color=False):
    """Load image for the camera.
    """

    self.img_path = img_path

    img = cv2.imread(
        self.img_path, cv2.IMREAD_COLOR)
    if convert_color:
      self._img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      self._img = img

    [self._height, self._width, _] = self._img.shape

    if gt_loc:
      self._img[gt_loc[0]-50:gt_loc[0]+50,
                gt_loc[1]-50: gt_loc[1]+50, 0] = 255
      self._img[gt_loc[0]-50: gt_loc[0]+50,
                gt_loc[1]-50: gt_loc[1]+50, 1:] = 0

  def look(self, lat, lng):
    """Look at lng,lat
    """
    self.lng = lng
    self.lat = lat

    self._calculateMap(self.fov, self.lat, self.lng,
                       self.output_image_h, self.output_image_w)

  def get_image(self):
    """Return the image in the FoV
    """
    image = cv2.remap(self._img, self.lat_map, self.lng_map,
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return image

  def get_map(self):
    """Return the map.
    """

    lat_map = (self.lat_map / self._width) * 360.0 - 180.0
    lng_map = ((self.lng_map / self._height) * 180.0 - 90.0) * -1.0

    return np.stack((lat_map, lng_map), axis=2)

  def get_pixel_map(self):
    """Return the pixel map.
    """
    lng_map = self.lng_map
    lat_map = self.lat_map

    return lat_map.astype(int), lng_map.astype(int)

  @staticmethod
  def find_nearest(array, value):
    """Find the nearest coordinates for given lat,lng
    """
    x_distances = array[:, :, 0] - value[0]
    y_distances = array[:, :, 1] - value[1]
    distances = np.sqrt(x_distances * x_distances + y_distances * y_distances)
    idx = distances.argmin()
    return np.unravel_index(idx, array[:, :, 0].shape)

  def get_image_coordinate_for(self, lat, lng):
    """Return coordinates for given lng,lat
    """
    coords = PanoramicCamera.find_nearest(self.get_map(), (lat, lng))

    # given lng,lat may not be in the FoV
    if coords[0] == 0 or coords[0] == self.output_image_w - 1 or coords[1] == 0 or coords[1] == self.output_image_h - 1:
      return None

    return coords

  def _calculateMap(self, FOV, THETA, PHI, height, width, RADIUS=128):
    """Calculate the pixel map.
    """
    equ_h = self._height
    equ_w = self._width
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / \
        np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / \
        np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
    xyz = np.zeros([height, width, 3], np.float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T

    lng = np.arcsin(xyz[:, 2] / RADIUS)
    lat = np.zeros([height * width], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    lat[idx1] = theta[idx1]
    lat[idx3] = theta[idx3] + np.pi
    lat[idx4] = theta[idx4] - np.pi

    lat = lat.reshape([height, width]) / np.pi * 180
    lng = -lng.reshape([height, width]) / np.pi * 180
    lat = lat / 180 * equ_cx + equ_cx
    lng = lng / 90 * equ_cy + equ_cy

    self.lng_map = lng.astype(np.float32)
    self.lat_map = lat.astype(np.float32)
