'''
Panoramic camera to generate perspective for equirectangular images.
adapted from here :  https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
'''
import cv2
import numpy as np
import os
from tqdm import tqdm

from scipy.spatial import cKDTree


class PanoramicCamera:
  def __init__(self, fov=90, output_image_shape=(400, 400)):
    '''Init the camera.
    '''
    self.fov = fov
    self.output_image_h = output_image_shape[0]
    self.output_image_w = output_image_shape[1]

    self.xlng_map = None
    self.ylat_map = None

    self.img_path = ''
    self._img = None

  def load_maps(self, cache_root):
    '''Loads precomputed maps.
    '''
    map_prefix = os.path.join(
        cache_root, 'maps')
    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    nodes = meta['nodes']

    self.masks = {}
    self.lng_maps = {}
    self.lat_maps = {}

    print('loading maps...')
    pbar = tqdm(nodes)
    for n in pbar:
      node = nodes[n]
      idx = node['idx']
      mask_file = os.path.join(map_prefix, '{}.mask.jpg'.format(idx))
      mask = cv2.imread(mask_file)
      maps_file = os.path.join(map_prefix, '{}.map.npy'.format(idx))
      maps_data = np.load(maps_file, allow_pickle=True)[()]
      self.lng_maps[idx] = maps_data['lng_map']
      self.lat_maps[idx] = maps_data['lat_map']
      self.masks[idx] = mask

  def look_fov(self, idx):
    '''Look at xlng, ylat
    '''
    self.xlng_map = self.lng_maps[idx]
    self.ylat_map = self.lat_maps[idx]
    self.mask = self.masks[idx]

  def load_img(self, img_path, gt_loc=None, convert_color=True):
    '''Load image for the camera.
    '''

    self.img_path = img_path

    img = cv2.imread(
        self.img_path, cv2.IMREAD_COLOR)
    if convert_color:
      self._img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      self._img = img

    [self._height, self._width, _] = self._img.shape

    if gt_loc:
      self._img[gt_loc[1]-50:gt_loc[1]+50,
                gt_loc[0]-50: gt_loc[0]+50, 0] = 255
      self._img[gt_loc[1]-50: gt_loc[1]+50,
                gt_loc[0]-50: gt_loc[0]+50, 1:] = 0

  def look(self, xlng, ylat):
    '''Look at xlng, ylat
    '''
    self.xlng = xlng
    self.ylat = ylat

    self._calculateMap(self.fov, self.xlng, self.ylat,
                       self.output_image_h, self.output_image_w)

  def get_image(self):
    '''Return the image in the FoV
    '''
    image = cv2.remap(self._img, self.xlng_map, self.ylat_map,
                      cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)
    return image

  def get_inverse_map(self):
    '''Returns inverse mapping.
    '''
    full_h, full_w = self._height, self._width
    h, w = self.output_image_h, self.output_image_w
    mapx_inverse = np.zeros((full_h, full_w))
    mapy_inverse = np.zeros((full_h, full_w))
    mapx, mapy = self.xlng_map, self.ylat_map

    data = []
    coords = []
    for j in range(w):
      for i in range(h):
        data.append([mapx[i, j], mapy[i, j]])
        coords.append((i, j))
    data = np.array(data)
    tree = cKDTree(data, leafsize=16, compact_nodes=True, balanced_tree=True)

    coords.append((0, 0))  # extra coords for failed neighbour search

    x = np.linspace(0.0, full_w, num=full_w, endpoint=False)
    y = np.linspace(0.0, full_h, num=full_h, endpoint=False)
    pts = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    distances, indices = tree.query(pts, k=5, p=2, distance_upper_bound=5.0)

    for (x, y), ds, idxs in zip(pts.astype(np.uint16), distances, indices):
      wsum_i = 0
      wsum_j = 0
      wsum = np.finfo(float).eps
      for d, idx in zip(ds, idxs):
        w = 1.0 / (d*d)
        wsum += w
        wsum_i += w*coords[idx][0]
        wsum_j += w*coords[idx][1]
      wsum_i /= wsum
      wsum_j /= wsum
      mapx_inverse[y, x] = wsum_j
      mapy_inverse[y, x] = wsum_i

    return mapx_inverse, mapy_inverse

  def get_map(self):
    '''Return the map.
    '''

    xlng_map = (self.xlng_map / self._width) * 360.0 - 180.0
    ylat_map = ((self.ylat_map / self._height) * 180.0 - 90.0) * -1.0
    return np.stack((xlng_map, ylat_map), axis=2)

  def get_pixel_map(self):
    '''Return the pixel map.
    '''
    xlng_map = self.xlng_map
    ylat_map = self.ylat_map

    return xlng_map, ylat_map

  @staticmethod
  def find_nearest(array, value):
    '''Find the nearest coordinates for given lat, lng
    '''
    y_distances = array[:, :, 1] - value[1]
    x_distances = array[:, :, 0] - value[0]

    distances = np.sqrt(x_distances * x_distances + y_distances * y_distances)
    idx = distances.argmin()
    return np.unravel_index(idx, array[:, :, 0].shape)

  def get_image_coordinate_for(self, xlng, ylat):
    '''Return coordinates for given lng, lat
    '''
    coords = PanoramicCamera.find_nearest(self.get_map(), (xlng, ylat))

    # given lng,lat may not be in the FoV
    if coords[0] == 0 or coords[0] == self.output_image_w - 1 or coords[1] == 0 or coords[1] == self.output_image_h - 1:
      return None

    return coords

  def _calculateMap(self, FOV, THETA, PHI, height, width, RADIUS=128.0):
    '''Calculate the pixel map.
    '''
    equ_h = self._height
    equ_w = self._width
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV  # 75 for Matterport
    hFOV = float(height) / width * wFOV  # 60 for Matterport

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

    ylat = np.arcsin(xyz[:, 2] / RADIUS)
    xlng = np.zeros([height * width], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    xlng[idx1] = theta[idx1]
    xlng[idx3] = theta[idx3] + np.pi
    xlng[idx4] = theta[idx4] - np.pi

    xlng = xlng.reshape([height, width]) / np.pi * 180
    ylat = -ylat.reshape([height, width]) / np.pi * 180
    xlng = xlng / 180 * equ_cx + equ_cx
    ylat = ylat / 90 * equ_cy + equ_cy

    self.xlng_map = xlng.astype(np.float32)
    self.ylat_map = ylat.astype(np.float32)
