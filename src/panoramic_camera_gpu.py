'''
Panoramic camera to generate perspective for equirectangular images.
adapted from here :  https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
'''
import cv2
import numpy as np
import torch


def deg2rad(theta):
  return torch.tensor([theta * np.pi / 180]).cuda()


def unravel_index(index, shape):
  out = []
  for dim in reversed(shape):
    out.append(index % dim)
    index = index // dim
  return tuple(reversed(out))


def rotation_matrix(axis, theta):
  """
  Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
  Return the rotation matrix associated with counterclockwise rotation about
  the given axis by theta radians.
  source: https://gist.github.com/fgolemo/94b5caf0e209a6e71ab0ce2d75ad3ed8
  """
  axis = axis / torch.sqrt(torch.dot(axis, axis))
  a = torch.cos(theta / 2.0)
  b, c, d = -axis * torch.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                       [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                       [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]).cuda()


class PanoramicCameraGPU:
  def __init__(self, fov=90,
               output_image_shape=(400, 400),
               use_tensor_image=False):
    """Init the camera.
    """
    self.fov = fov
    self.output_image_h = output_image_shape[0]
    self.output_image_w = output_image_shape[1]
    self.use_tensor_image = use_tensor_image

    self.lng_map = None
    self.lat_map = None

    self.img_path = ''
    self._img = None

  def load_img(self, img_path, gt_loc=None, convert_color=True):
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

    img = torch.tensor(self._img).float().permute(2, 0, 1).unsqueeze(0).cuda()
    mapping = torch.stack((self.lat_map, self.lng_map), axis=2).unsqueeze(0)

    w = img.shape[3]
    h = img.shape[2]
    mapping[:, :, :, 0] -= w/2
    mapping[:, :, :, 0] /= w/2
    mapping[:, :, :, 1] -= h/2
    mapping[:, :, :, 1] /= h/2

    image = torch.nn.functional.grid_sample(img, mapping,
                                            align_corners=True,
                                            padding_mode='border').squeeze(0).permute(1, 2, 0)

    if self.use_tensor_image:
      return image

    image = image.permute(2, 0, 1).cpu()
    np_image = image.data.numpy().transpose(1, 2, 0)
    np_image = np_image.astype(np.uint8)
    return np_image

  def get_map(self, use_tensor=False):
    lat_map = (self.lat_map / self._width) * 360.0 - 180.0
    lng_map = ((self.lng_map / self._height) * 180.0 - 90.0) * -1.0

    if use_tensor:
      return torch.stack((lat_map, lng_map), axis=2)
    return torch.stack((lat_map, lng_map), axis=2).detach().cpu().numpy()

  def get_pixel_map(self):
    """Return the pixel map.
    """
    lng_map = self.lng_map
    lat_map = self.lat_map

    return lat_map.long(), lng_map.long()

  @staticmethod
  def find_nearest(array, value):
    """Find the nearest coordinates for given lat,lng
    """
    x_distances = array[:, :, 0] - value[0]
    y_distances = array[:, :, 1] - value[1]
    distances = torch.sqrt(x_distances * x_distances +
                           y_distances * y_distances)
    idx = distances.argmin()
    return unravel_index(idx, array[:, :, 0].shape)

  def get_image_coordinate_for(self, lat, lng):
    """Return coordinates for given lng,lat
    """
    coords = PanoramicCameraGPU.find_nearest(
        self.get_map(use_tensor=True), (lat, lng))

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
    w_len = 2 * RADIUS * torch.sin(deg2rad(wFOV / 2.0)) / \
        torch.sin(deg2rad(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * torch.sin(deg2rad(hFOV / 2.0)) / \
        torch.sin(deg2rad(hangle))
    h_interval = h_len / (height - 1)
    x_map = torch.zeros([height, width]).cuda() + RADIUS
    y_map = ((torch.arange(0, width).cuda() - c_x) *
             w_interval).repeat(height, 1)
    z_map = -((torch.arange(0, height).cuda() - c_y) *
              h_interval).repeat(width, 1).transpose(1, 0)
    D = torch.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2).cuda()
    xyz = torch.zeros([height, width, 3]).cuda()
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = torch.tensor([[0.0, 1.0, 0.0]]).cuda().transpose(1, 0)
    z_axis = torch.tensor([0.0, 0.0, 1.0]).cuda()

    R1 = rotation_matrix(z_axis, deg2rad(THETA))
    R2 = rotation_matrix(R1.mm(y_axis).transpose(
        1, 0).reshape(3), deg2rad(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = R1.mm(xyz)
    xyz = R2.mm(xyz).transpose(1, 0)

    lng = torch.asin(xyz[:, 2] / RADIUS)
    lat = torch.zeros([height * width]).cuda()
    theta = torch.atan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1.long()) * idx2.long()).bool()
    idx4 = ((1 - idx1.long()) * (1 - idx2.long())).bool()

    lat[idx1] = theta[idx1]
    lat[idx3] = theta[idx3] + np.pi
    lat[idx4] = theta[idx4] - np.pi

    lat = lat.reshape([height, width]) / np.pi * 180
    lng = -lng.reshape([height, width]) / np.pi * 180
    lat = lat / 180 * equ_cx + equ_cx
    lng = lng / 90 * equ_cy + equ_cy

    self.lng_map = lng.float()
    self.lat_map = lat.float()
