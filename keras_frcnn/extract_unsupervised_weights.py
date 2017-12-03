from keras.applications.mobilenet import MobileNet
import sys
sys.path.append('/lfs/1/thaonguyen/PuzzleSolving/')
from train_puzzle import PuzzleModel

model = PuzzleModel(input_shape=(76,76,27))
new_model = MobileNet(input_shape=None, include_top=False, weights=None)
model.load_weights('/lfs/1/thaonguyen/PuzzleSolving/WEIGHTS/weights-best.hdf5')

mbnet_layer = model.get_layer('mobilenet_1.00_76')
for layer in new_model.layers[1:]:
    layer.set_weights(mbnet_layer.get_layer(layer.name).get_weights())
#print mbnet_layer.summary() 
new_model.save_weights('mobilenet-weights-best.hdf5')