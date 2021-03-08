'''
Cached Panoramic camera to generate perspective for equirectangular images.
'''
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from pprint import pprint


class CachedPanoramicCamera():
  def __init__(self, cache_root,
               fov=90,
               output_image_shape=(400, 400),
               height=2276,
               width=4552):
    '''Init the camera.
    '''
    super().__init__()
    self._height, self._width = height, width
    self.fov = fov
    self.output_image_h = output_image_shape[0]
    self.output_image_w = output_image_shape[1]

    self.xlng_map = None
    self.ylat_map = None

    self.img_path = ''
    self._img = None
    self.cache_root = cache_root
    self.meta_file = os.path.join(self.cache_root, 'meta.npy')
    meta = np.load(self.meta_file, allow_pickle=True)[()]
    self.nodes = meta['nodes']
    self.edges = meta['edges']
    self.graph = meta['graph']
    self.paths = meta['paths']
    self.distances = meta['distances']
    self.fovs = None
    self.fov = None
    self.masks = {}
    self.lng_maps = {}
    self.lat_maps = {}

  def __copy__(self):
    obj = type(self).__new__(self.__class__)
    obj.__dict__.update(self.__dict__)
    return obj

  def set_pano(self, pano):
    self.pano = pano
    self._img = None
    self.img_path = ''

  def load_fovs(self,
                verbose=False):
    '''Loads precropped fovs.
    '''
    if self.pano == '':
      print('you need to load image or set pano')
      quit(0)

    fov_prefix = os.path.join(
        self.cache_root, 'fovs', '{}'.format(self.pano))

    self.fovs = {}
    if verbose:
      print('loading fovs...')
      pbar = tqdm(self.nodes)
    else:
      pbar = self.nodes
    for n in pbar:
      node = self.nodes[n]
      idx = node['idx']
      fov_file = fov_prefix + '.{}.jpg'.format(idx)

      if not os.path.exists(fov_file):
        print('file missing:', fov_file)
        quit(0)

      fov = np.array(Image.open(fov_file))
      self.fovs[idx] = fov

  def load_masks(self):
    '''Loads precomputed masks.
    '''
    map_prefix = os.path.join(
        self.cache_root, 'maps')

    self.masks = {}

    print('loading masks...')
    pbar = tqdm(self.nodes)
    for n in pbar:
      node = self.nodes[n]
      idx = node['idx']
      mask_file = os.path.join(map_prefix, '{}.mask.jpg'.format(idx))
      mask = np.array(Image.open(mask_file))
      self.masks[idx] = mask

  def load_maps(self):
    '''Loads precomputed maps.
    '''
    map_prefix = os.path.join(
        self.cache_root, 'maps')

    print('loading maps...')
    pbar = tqdm(self.nodes)
    for n in pbar:
      node = self.nodes[n]
      idx = node['idx']
      maps_file = os.path.join(map_prefix, '{}.map.npy'.format(idx))
      maps_data = np.load(maps_file, allow_pickle=True)[()]
      self.lng_maps[idx] = maps_data['lng_map']
      self.lat_maps[idx] = maps_data['lat_map']

  def look_fov(self, idx):
    '''Look at xlng, ylat
    '''
    self.idx = idx
    self.xlng_map = self.lng_maps.get(idx, None)
    self.ylat_map = self.lat_maps.get(idx, None)
    self.mask = self.masks.get(idx, None)
    self.x = self.nodes[idx]['x']
    self.y = self.nodes[idx]['y']
    self.lng = self.nodes[idx]['lng']
    self.lat = self.nodes[idx]['lat']
    if self.fovs is not None:
      self.fov = self.fovs[idx]

  def get_neighbors(self):
    node = self.nodes[self.idx]

    neighbors = [self.nodes[n] for n in node['neighbor2dir'].keys()]
    return neighbors

  def get_current(self):
    return self.x, self.y

  def get_mask(self):
    return self.mask

  def load_img(self, img_path, gt_loc=None):
    '''Load image for the camera.
    '''
    self.img_path = img_path

    self._img = np.array(Image.open(self.img_path))

    [self._height, self._width, _] = self._img.shape

    if gt_loc:
      self._img[gt_loc[1]-50:gt_loc[1]+50,
                gt_loc[0]-50: gt_loc[0]+50, 0] = 255
      self._img[gt_loc[1]-50: gt_loc[1]+50,
                gt_loc[0]-50: gt_loc[0]+50, 1:] = 0
    self.pano = self.img_path.split('/')[-1].split('.')[0]

  def look(self, xlng, ylat):
    '''Look at xlng, ylat
    '''
    raise NotImplementedError('this camera does not implement look()')

  def get_image(self):
    '''Return the image in the FoV
    '''
    if self.fov is not None:
      return self.fov
    print('load cached FoVs.')
    quit(1)

  def get_inverse_map(self):
    '''Returns inverse mapping.
    '''
    raise NotImplementedError(
        'this camera does not implement get_inverse_map()')

  def get_map(self):
    '''Return the map.
    '''
    xlng_map = (self.xlng_map / self._width) * 360.0 - 180.0  # TODO: check
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
    coords = CachedPanoramicCamera.find_nearest(self.get_map(), (xlng, ylat))

    # given lng,lat may not be in the FoV
    if coords[0] == 0 or coords[0] == self.output_image_w - 1 or coords[1] == 0 or coords[1] == self.output_image_h - 1:
      return None

    return coords

  def _calculateMap(self, FOV, THETA, PHI, height, width, RADIUS=128.0):
    '''Calculate the pixel map.
    '''
    raise NotImplementedError('this camera does not implement _calculateMap()')
