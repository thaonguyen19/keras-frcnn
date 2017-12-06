from optparse import OptionParser
import os
from measure_map import map_main

parser = OptionParser()
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("--weights_folder", dest="weights_folder", help="Path to all the weights")
(options, args) = parser.parse_args()

max_map = float('-inf')
max_file = None
config_output_filename = 'config.pickle'
img_path = options.test_path

for weights_filename in os.listdir(options.weights_folder):
        weights_filename = os.path.join(options.weights_folder, weights_filename)
	map_score = map_main(config_output_filename, img_path, weights_filename, num_rois=32)
	if map_score > max_map:
		max_map = max_score
		max_file = weights_filename
		print "FOUND NEW mAP SCORE: %f with file %s" % (map_score, weights_filename)

print "FINAL: ", max_map, " ", weights_filename
