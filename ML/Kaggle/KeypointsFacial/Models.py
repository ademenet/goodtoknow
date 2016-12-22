import lasagne


def lenet5(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=((2, 2), (2, 2), (2, 2))):
	lenet5_do = {}
	lenet5_do['input1'] = lasagne.layers.InputLayer((None, 1, 96, 96))
	lenet5_do['conv1'] = lasagne.layers.Conv2DLayer(lenet5_do['input1'], num_filters=32, filter_size=(3, 3))
	lenet5_do['pool1'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv1'], pool_size=psize[0])
	lenet5_do['dropout1'] = lasagne.layers.DropoutLayer(lenet5_do['pool1'], p=dropout[0])
	lenet5_do['conv2'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout1'], num_filters=64, filter_size=(2, 2))
	lenet5_do['pool2'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv2'], pool_size=psize[1])
	lenet5_do['dropout2'] = lasagne.layers.DropoutLayer(lenet5_do['pool2'], p=dropout[1])
	lenet5_do['conv3'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout2'], num_filters=128, filter_size=(2, 2))
	lenet5_do['pool3'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv3'], pool_size=psize[2])
	lenet5_do['dropout3'] = lasagne.layers.DropoutLayer(lenet5_do['pool3'], p=dropout[2])
	lenet5_do['hidden4'] = lasagne.layers.DenseLayer(lenet5_do['dropout3'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['dropout4'] = lasagne.layers.DropoutLayer(lenet5_do['hidden4'], p=dropout[3])
	lenet5_do['hidden5'] = lasagne.layers.DenseLayer(lenet5_do['hidden4'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['output'] = lasagne.layers.DenseLayer(lenet5_do['hidden5'], num_units=output_unit, nonlinearity=None)
	return lenet5_do['output']


def lenet5_2(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=((2, 2), (2, 2), (2, 2))):
	lenet5_do = {}
	lenet5_do['input1'] = lasagne.layers.InputLayer((None, 1, 96, 96))
	lenet5_do['conv1'] = lasagne.layers.Conv2DLayer(lenet5_do['input1'], num_filters=32, filter_size=(3, 3))
	lenet5_do['pool1'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv1'], pool_size=psize[0])
	lenet5_do['dropout1'] = lasagne.layers.DropoutLayer(lenet5_do['pool1'], p=dropout[0])
	lenet5_do['conv2'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout1'], num_filters=64, filter_size=(2, 2))
	lenet5_do['pool2'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv2'], pool_size=psize[1])
	lenet5_do['dropout2'] = lasagne.layers.DropoutLayer(lenet5_do['pool2'], p=dropout[1])
	lenet5_do['conv3'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout2'], num_filters=128, filter_size=(2, 2))
	lenet5_do['pool3'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv3'], pool_size=psize[2])
	lenet5_do['dropout3'] = lasagne.layers.DropoutLayer(lenet5_do['pool3'], p=dropout[2])
	lenet5_do['hidden4'] = lasagne.layers.DenseLayer(lenet5_do['dropout3'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['dropout4'] = lasagne.layers.DropoutLayer(lenet5_do['hidden4'], p=dropout[3])
	lenet5_do['hidden5'] = lasagne.layers.DenseLayer(lenet5_do['hidden4'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['output'] = lasagne.layers.DenseLayer(lenet5_do['hidden5'], num_units=output_unit, nonlinearity=None)
	return lenet5_do['output']

def lenet5_3(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=((2, 2), (2, 2), (2, 2))):
	lenet5_do = {}
	lenet5_do['input1'] = lasagne.layers.InputLayer((None, 1, 96, 96))
	lenet5_do['conv1'] = lasagne.layers.Conv2DLayer(lenet5_do['input1'], num_filters=128, filter_size=(3, 3))
	lenet5_do['pool1'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv1'], pool_size=psize[0])
	lenet5_do['dropout1'] = lasagne.layers.DropoutLayer(lenet5_do['pool1'], p=dropout[0])
	lenet5_do['conv2'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout1'], num_filters=216, filter_size=(2, 2))
	lenet5_do['pool2'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv2'], pool_size=psize[1])
	lenet5_do['dropout2'] = lasagne.layers.DropoutLayer(lenet5_do['pool2'], p=dropout[1])
	lenet5_do['conv3'] = lasagne.layers.Conv2DLayer(lenet5_do['dropout2'], num_filters=514, filter_size=(2, 2))
	lenet5_do['pool3'] = lasagne.layers.MaxPool2DLayer(lenet5_do['conv3'], pool_size=psize[2])
	lenet5_do['dropout3'] = lasagne.layers.DropoutLayer(lenet5_do['pool3'], p=dropout[2])
	lenet5_do['hidden4'] = lasagne.layers.DenseLayer(lenet5_do['dropout3'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['dropout4'] = lasagne.layers.DropoutLayer(lenet5_do['hidden4'], p=dropout[3])
	lenet5_do['hidden5'] = lasagne.layers.DenseLayer(lenet5_do['hidden4'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	lenet5_do['output'] = lasagne.layers.DenseLayer(lenet5_do['hidden5'], num_units=output_unit, nonlinearity=None)
	return lenet5_do['output']


def model1(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5)):
	input1 = lasagne.layers.InputLayer((None, 1, 96, 96))
	conv1 = lasagne.layers.Conv2DLayer(input1, num_filters=32, filter_size=(3, 3))
	conv2 = lasagne.layers.Conv2DLayer(conv1, num_filters=32, filter_size=(3, 3))
	pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
	dropout1 = lasagne.layers.DropoutLayer(pool1, p=dropout[0])
	conv3 = lasagne.layers.Conv2DLayer(dropout1, num_filters=64, filter_size=(2, 2))
	conv4 = lasagne.layers.Conv2DLayer(conv3, num_filters=64, filter_size=(2, 2))
	pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))
	dropout2 = lasagne.layers.DropoutLayer(pool2, p=dropout[1])
	conv5 = lasagne.layers.Conv2DLayer(dropout2, num_filters=128, filter_size=(2, 2))
	conv6 = lasagne.layers.Conv2DLayer(conv5, num_filters=128, filter_size=(2, 2))
	pool3 = lasagne.layers.MaxPool2DLayer(conv6, pool_size=(2, 2))
	dropout3 = lasagne.layers.DropoutLayer(pool3, p=dropout[2])
	hidden1 = lasagne.layers.DenseLayer(dropout3, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	dropout4 = lasagne.layers.DropoutLayer(hidden1, p=dropout[3])
	hidden2 = lasagne.layers.DenseLayer(dropout4, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	output = lasagne.layers.DenseLayer(hidden2, num_units=output_unit, nonlinearity=None)
	return output


def model2(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5)):
	input1 = lasagne.layers.InputLayer((None, 1, 96, 96))
	conv1 = lasagne.layers.Conv2DLayer(input1, num_filters=64, filter_size=(3, 3))
	conv2 = lasagne.layers.Conv2DLayer(conv1, num_filters=64, filter_size=(3, 3))
	pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
	dropout1 = lasagne.layers.DropoutLayer(pool1, p=dropout[0])
	conv3 = lasagne.layers.Conv2DLayer(dropout1, num_filters=128, filter_size=(2, 2))
	conv4 = lasagne.layers.Conv2DLayer(conv3, num_filters=128, filter_size=(2, 2))
	pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))
	dropout2 = lasagne.layers.DropoutLayer(pool2, p=dropout[1])
	conv5 = lasagne.layers.Conv2DLayer(dropout2, num_filters=256, filter_size=(2, 2))
	conv6 = lasagne.layers.Conv2DLayer(conv5, num_filters=256, filter_size=(2, 2))
	pool3 = lasagne.layers.MaxPool2DLayer(conv6, pool_size=(2, 2))
	dropout3 = lasagne.layers.DropoutLayer(pool3, p=dropout[2])
	hidden1 = lasagne.layers.DenseLayer(dropout3, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	dropout4 = lasagne.layers.DropoutLayer(hidden1, p=dropout[3])
	hidden2 = lasagne.layers.DenseLayer(dropout4, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	output = lasagne.layers.DenseLayer(hidden2, num_units=output_unit, nonlinearity=None)
	return output


def model3(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=None):
	input1 = lasagne.layers.InputLayer((None, 1, 96, 96))
	conv1 = lasagne.layers.Conv2DLayer(input1, num_filters=128, filter_size=(3, 3))
	conv2 = lasagne.layers.Conv2DLayer(conv1, num_filters=128, filter_size=(3, 3))
	pool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
	dropout1 = lasagne.layers.DropoutLayer(pool1, p=dropout[0])
	conv3 = lasagne.layers.Conv2DLayer(dropout1, num_filters=256, filter_size=(2, 2))
	conv4 = lasagne.layers.Conv2DLayer(conv3, num_filters=256, filter_size=(2, 2))
	pool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))
	dropout2 = lasagne.layers.DropoutLayer(pool2, p=dropout[1])
	conv5 = lasagne.layers.Conv2DLayer(dropout2, num_filters=512, filter_size=(2, 2))
	conv6 = lasagne.layers.Conv2DLayer(conv5, num_filters=512, filter_size=(2, 2))
	pool3 = lasagne.layers.MaxPool2DLayer(conv6, pool_size=(3, 3))
	dropout3 = lasagne.layers.DropoutLayer(pool3, p=dropout[2])
	hidden1 = lasagne.layers.DenseLayer(dropout3, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	dropout4 = lasagne.layers.DropoutLayer(hidden1, p=dropout[3])
	hidden2 = lasagne.layers.DenseLayer(dropout4, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
	output = lasagne.layers.DenseLayer(hidden2, num_units=output_unit, nonlinearity=None)
	return output


def lenet(output_unit=30, dropout=None):
	le_net_like = {}
	le_net_like['input1'] = lasagne.layers.InputLayer((None, 1, 96, 96))
	le_net_like['conv1'] = lasagne.layers.Conv2DLayer(le_net_like['input1'], num_filters=32, filter_size=(3, 3))
	le_net_like['pool1'] = lasagne.layers.MaxPool2DLayer(le_net_like['conv1'], pool_size=(2, 2))
	le_net_like['conv2'] = lasagne.layers.Conv2DLayer(le_net_like['pool1'], num_filters=64, filter_size=(2, 2))
	le_net_like['pool2'] = lasagne.layers.MaxPool2DLayer(le_net_like['conv2'], pool_size=(2, 2))
	le_net_like['conv3'] = lasagne.layers.Conv2DLayer(le_net_like['pool2'], num_filters=128, filter_size=(2, 2))
	le_net_like['pool3'] = lasagne.layers.MaxPool2DLayer(le_net_like['conv3'], pool_size=(2, 2))
	le_net_like['hidden4'] = lasagne.layers.DenseLayer(le_net_like['pool3'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	le_net_like['hidden5'] = lasagne.layers.DenseLayer(le_net_like['hidden4'], num_units=500, nonlinearity=lasagne.nonlinearities.rectify)
	le_net_like['output'] = lasagne.layers.DenseLayer(le_net_like['hidden5'], num_units=output_unit, nonlinearity=None)
	return le_net_like['output']


def vgg16_1(output_unit=30, dropout=None):
	vgg = {}
	vgg['input'] = lasagne.layers.InputLayer((None, 1, 96, 96))

	vgg['conv1'] = lasagne.layers.Conv2DLayer(vgg['input'], num_filters=4, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv2'] = lasagne.layers.Conv2DLayer(vgg['conv1'], num_filters=4, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['pool1'] = lasagne.layers.MaxPool2DLayer(vgg['conv2'], pool_size=(2, 2))
	
	vgg['conv3'] = lasagne.layers.Conv2DLayer(vgg['pool1'], num_filters=8, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv4'] = lasagne.layers.Conv2DLayer(vgg['conv3'], num_filters=8, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['pool2'] = lasagne.layers.MaxPool2DLayer(vgg['conv4'], pool_size=(2, 2))
	
	vgg['conv5'] = lasagne.layers.Conv2DLayer(vgg['pool2'], num_filters=16, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv6'] = lasagne.layers.Conv2DLayer(vgg['conv5'], num_filters=16, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv7'] = lasagne.layers.Conv2DLayer(vgg['conv6'], num_filters=16, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['pool3'] = lasagne.layers.MaxPool2DLayer(vgg['conv7'], pool_size=(2, 2))
	
	vgg['conv8'] = lasagne.layers.Conv2DLayer(vgg['pool3'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv9'] = lasagne.layers.Conv2DLayer(vgg['conv8'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv10'] = lasagne.layers.Conv2DLayer(vgg['conv9'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['pool4'] = lasagne.layers.MaxPool2DLayer(vgg['conv10'], pool_size=(2, 2))
	
	vgg['conv11'] = lasagne.layers.Conv2DLayer(vgg['pool4'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv12'] = lasagne.layers.Conv2DLayer(vgg['conv11'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['conv13'] = lasagne.layers.Conv2DLayer(vgg['conv12'], num_filters=32, filter_size=(3, 3), pad=1, flip_filters=False, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['pool5'] = lasagne.layers.MaxPool2DLayer(vgg['conv10'], pool_size=(2, 2))
	
	vgg['fc1'] = lasagne.layers.DenseLayer(vgg['pool5'], num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['fc2'] = lasagne.layers.DenseLayer(vgg['fc1'], num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	vgg['fc3'] = lasagne.layers.DenseLayer(vgg['fc2'], num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
	
	vgg['output'] = lasagne.layers.DenseLayer(vgg['fc3'], num_units=output_unit)
	return vgg['output']


# Inspired by: https://github.com/ishay2b/VanillaCNN/blob/master/python/VanillaNoteBook.ipynb
def vanillamodel(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=None):
	l = lasagne.layers.InputLayer((None, 1, 96, 96))
	l = lasagne.layers.Conv2DLayer(l, num_filters=16, filter_size=(5, 5), pad=2, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.Conv2DLayer(l, num_filters=48, filter_size=(3, 3), pad=1, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(3, 3), pad=0, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), pad=1, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.DenseLayer(l, num_units=output_unit)
	return l

def BigConv(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=None):
	l = lasagne.layers.InputLayer((None, 1, 96, 96))

	l = lasagne.layers.Conv2DLayer(l, num_filters=16, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify) # 94
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.1)

	l = lasagne.layers.Conv2DLayer(l, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify) # 46 
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.1)

	l = lasagne.layers.Conv2DLayer(l, num_filters=48, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 22
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.2)

	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 10
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.3)

	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 4
	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 3
	l = lasagne.layers.DropoutLayer(l, p=0.4)

	l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.DropoutLayer(l, p=0.5)
	l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

	l = lasagne.layers.DenseLayer(l, num_units=output_unit, nonlinearity=None)
	return l

def BigConv2(output_unit=30, dropout=(0.1, 0.2, 0.3, 0.5), psize=None):
	l = lasagne.layers.InputLayer((None, 1, 96, 96))

	l = lasagne.layers.Conv2DLayer(l, num_filters=16, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify) # 94
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.1)

	l = lasagne.layers.Conv2DLayer(l, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify) # 46 
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.1)

	l = lasagne.layers.Conv2DLayer(l, num_filters=48, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 22
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.2)

	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 10
	l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))
	l = lasagne.layers.DropoutLayer(l, p=0.3)

	l = lasagne.layers.Conv2DLayer(l, num_filters=64, filter_size=(2, 2), nonlinearity=lasagne.nonlinearities.rectify) # 4
	l = lasagne.layers.DropoutLayer(l, p=0.4)

	l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
	l = lasagne.layers.DropoutLayer(l, p=0.5)
	l = lasagne.layers.DenseLayer(l, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

	l = lasagne.layers.DenseLayer(l, num_units=output_unit, nonlinearity=None)
	return l


models = {
	'lenet': lenet,
	'lenet5': lenet5,
	'lenet5_2': lenet5_2,
	'lenet5_3': lenet5_3,
	'vgg16_1': vgg16_1,
	'model1': model1,
	'model2': model2,
	'model3': model3,
	'vanillamodel': vanillamodel,
	'BigConv': BigConv,
	'BigConv2': BigConv2,
}
