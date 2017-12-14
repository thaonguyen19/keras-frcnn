from sklearn.neighbors import NearestNeighbors
from keras import backend as K
import cv2
from PIL import Image
import random
import numpy as np
import pickle
import sys
import config
from optparse import OptionParser
from pascal_voc_parser import get_data
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import img_to_array

K.set_learning_phase(1)
config_output_filename = 'nearest_neighbor_puzzle.pickle'
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--input_weight", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

C = config.Config()

C.num_rois = int(options.num_rois)
C.network = 'mobilenet'
import mobilenet as nn

C.base_net_weights = options.input_weight_path

all_imgs, classes_count, class_mapping = get_data(options.train_path)
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))
n_bbox = 0
widths, heights = 0, 0
#no data augmentation
#data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
#data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
output = nn.nn_base(input_tensor=img_input, trainable=False)
base_model = Model(inputs=img_input, outputs=output)
base_model.load_weights(C.base_net_weights, by_name=True)

def load_image(filename, dim_w, dim_h):
    img = Image.open(filename).convert('RGB')
    img = np.asarray(img)
    resized = False
    '''
    if img.shape[0] < dim_w:
        resized = True
        print dim_w, img.shape
        img = img.resize((dim_w, img.shape[1]), Image.ANTIALIAS)    
    if img.shape[1] < dim_h:
        if resized:
            img = img.resize((dim_w, dim_h), Image.ANTIALIAS)
        else:
            img = img.resize((img.shape[0], dim_h), Image.ANTIALIAS)
    '''
    return img

def crop_and_save(filename, x1, x2, y1, y2, output_filename):
    img = Image.open(filename).convert('RGB')
    img = np.asarray(img)
    patch = img[x1:x2, y1:y2, :]
    img = Image.fromarray(patch) 
    img.save(output_filename)   

def extract_feature(img_data, all_nn_fit_data, fit_data_info):
	bboxes = img_data['bboxes']
	path = img_data['filepath']
	#global n_bbox, widths, heights
	for bbox in bboxes:
		img = load_image(path, C.im_size, C.im_size)
		x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
		if (x2 - x1) * (y2 - y1) < 10000:
			continue
		#n_bbox += 1
		#widths += (x2-x1)
		#heights += (y2-y1)
		#print x2-x1, y2-y1
		bbox_crop = img[x1:x2, y1:y2, :]
		bbox_crop = np.expand_dims(bbox_crop, axis=0) #add 1 as first axis for batch size dimension
		bbox_crop = np.resize(bbox_crop, (1, 200,200, 3))
		feature = base_model.predict(bbox_crop)
		feature_arr = np.squeeze(feature, axis=0)
		norm_feature = feature_arr/ np.linalg.norm(feature_arr)
		all_nn_fit_data.append(norm_feature)
		fit_data_info.append((path, bbox['class'], x1, x2, y1, y2))

all_nn_fit_data = []
fit_data_info = []
print "START COMPUTING FEATURES FROM TRAIN SET"
for img_data in train_imgs:
	extract_feature(img_data, all_nn_fit_data, fit_data_info)
#print "TOTAL BOXES: ", n_bbox
#print "MEAN WIDTH: ", float(widths)/n_bbox
#print "MEAN HEIGHT: ", float(heights)/n_bbox

#with open(config_output_filename, 'wb') as config_f:
#	pickle.dump(neigh, config_f)
#	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

count = 0
top_k = 10
top_n_pixels = 200*200

for img_data in val_imgs:
	if count == 10:
		break
	all_inner_prod = []
	query_feature = []
	query_data = []
	extract_feature(img_data, query_feature, query_data)
	if len(query_feature)==0:
		continue
	count += 1
	print "QUERY IMG: ", query_data[0]
	path, _, x1, x2, y1, y2 = query_data[0]
	crop_and_save(path, x1, x2, y1, y2, 'query-'+str(count)+'.png')
	for i in range(len(all_nn_fit_data)):
		retrieve_feature = all_nn_fit_data[i]
		norm_inner_prod = np.inner(query_feature[0].flatten(), retrieve_feature.flatten())
		#inner_prod = inner_prod.flatten()
		#inner_prod = inner_prod[np.argsort(inner_prod)[-top_n_pixels:]]
		#norm_inner_prod = np.sum(inner_prod)#*1.0/np.prod(inner_prod.shape)
		all_inner_prod.append(norm_inner_prod)
	print "RETRIEVING: "
	top_ind = sorted(range(len(all_inner_prod)), key=lambda x: all_inner_prod[x])
	top_ind = top_ind[-top_k:]
	for ind in range(len(top_ind)):
		print fit_data_info[top_ind[ind]]
		path, _, x1, x2, y1, y2 = fit_data_info[top_ind[ind]]
        	crop_and_save(path, x1, x2, y1, y2, 'retrieve-'+str(count)+'-'+str(ind)+'.png')
