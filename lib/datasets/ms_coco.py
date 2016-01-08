# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets, math
import datasets.ms_coco
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess

class ms_coco(datasets.imdb):
    def __init__(self, image_set, year = 2014, coco_path=None):
        datasets.imdb.__init__(self, 'coco_' + image_set + year)
        self._year = year
        self._image_set = image_set
        self._coco_name = image_set + year
        self._coco_path = self._get_default_path() if coco_path is None \
                            else coco_path
        self._COCO = self._load_coco_json()

        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [cat['name'] for cat in cats])
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self._validate_image_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'top_k'    : 2000}

        assert os.path.exists(self._coco_path), \
                'VOCdevkit path does not exist: {}'.format(self._coco_path)

    def _validate_image_index(self):
        # training requirs at least one object in the image, remove those without
        if int(self._year) == 2007 or self._image_set != 'test':
            roidb = self.gt_roidb(False) #disable caching, because we are going to rebuild index

            image_index = []
            for rois, index in zip(roidb, self._image_index):
                if rois['gt_overlaps'].size != 0:
                    image_index.append(index)
            self._image_index = image_index


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def _get_image_filename(self, index):
        return self._COCO.loadImgs(index)[0]['file_name']

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._coco_path, 'images', self._coco_name, self._get_image_filename(index))
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_index = self._COCO.getImgIds()
        return image_index

    def _load_coco_json(self):
        import sys
        sys.path.append(os.path.join(self._coco_path, 'PythonAPI'))
        try:
            from pycocotools.coco import COCO
        except:
            raise Exception("can't find coco API in the coco path: %s" % self._coco_path)

        ann_file = os.path.join(self._coco_path, 'annotations', 'instances_' + self._coco_name + '.json')
        return COCO(ann_file)

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'coco')

    def gt_roidb(self, caching = True):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if caching and os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self.image_index]

        if caching:
            with open(cache_file, 'wb') as fid:
                cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        images, boxes = self._load_v73_ss_boxes(filename)

        box_dict = {}
        for img, box_tmp in zip(images, boxes):
            box_dict[img] = box_tmp[:, (1, 0, 3, 2)] - 1

        box_list = []
        for index in self._image_index:
            file_name = self._get_image_filename(index)
            box_list.append(box_dict[file_name])

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    #http://stackoverflow.com/a/28259682
    #ss file is saved by matlab with -v7.3
    #two variables: boxes and images
    # images: cell array of filenames
    # boxes: cell array of boxes:
    #   each cell is a n_boxes * 4 (y1,x1,y2,x2 in Selective Search Code) matrix
    def _load_v73_ss_boxes(self, path):
        import h5py

        with h5py.File(path) as reader:

            images = []
            for column in reader['images']:
                row_data = []
                for row_number in range(len(column)):
                    row_data.append(''.join(map(unichr, reader[column[row_number]][:])))
                images.append(row_data[0])

            boxes = []
            for column in reader['boxes']:
                row_data = []
                for row_number in range(len(column)):
                    box_tmp = reader[column[row_number]][:]
                    row_data.append(np.transpose(box_tmp))
                boxes = row_data

        return images, boxes

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_coco_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        objs = [obj for obj in objs if obj['area'] > 0]
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = round(obj['bbox'][0])
            y1 = round(obj['bbox'][1])
            x2 = x1 + math.ceil(obj['bbox'][2]) - 1
            y2 = y1 + math.ceil(obj['bbox'][3]) - 1
            cls = self._class_to_ind[self._COCO.loadCats(obj['category_id'])[0]['name']]
            boxes[ix, :] = [x1, y1, x2, y2]
            assert (boxes[ix,:]>=0).all()
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        raise Exception("not implemented")
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.ms_coco('val', '2014')
    res = d.roidb
    from IPython import embed; embed()
