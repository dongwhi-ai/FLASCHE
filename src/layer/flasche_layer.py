import numpy as np

########################### Level 1 class: Layer ###########################
class FLASCHELayer():
    def __init__(self):
        pass

########################### Level 2 classes: ConvLayer, Top_ConvLayer, ZeroPaddingLayer, ... ###########################
class ConvLayer(FLASCHELayer):
    def __init__(self, kernel_size, strides, input_max):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.input_max = input_max

class Top_ConvLayer(FLASCHELayer):
    def __init__(self, kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_weight_based_threshold = use_weight_based_threshold
        self.input_max = input_max
        self.all_ctcam_creation = all_ctcam_creation
        self.top_num = top_num
        self.top_thresholds = top_thresholds
        self.filter_weight_min = filter_weight_min
        self.dynamic_kernel_order = dynamic_kernel_order
        kernel_y, kernel_x = kernel_size
        bits = kernel_y * kernel_x
        self.bits = bits
        self.weights = np.array([2**i for i in range(bits)])[::-1]
        self.top_quantization_level = len(top_thresholds) + 1
        

class ZeroPaddingLayer(FLASCHELayer):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

class MaxPooling2DLayer(FLASCHELayer):
    def __init__(self, pool_size, strides):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides

class NormalizationLayer(FLASCHELayer):
    def __init__(self, stretching, new_max, old_min, old_max, normalize_per_group=False, normalize_per_image=True, adaptability='Weak'):
        super().__init__()
        self.stretching = stretching
        self.new_max = new_max
        self.old_min = old_min
        self.old_max = old_max
        if (normalize_per_image):
            normalize_per_group = False
        self.normalize_per_group = normalize_per_group
        self.normalize_per_image = normalize_per_image
        self.adaptability = adaptability

class ScoringLayer(FLASCHELayer):
    def __init__(self, strides, exclude_zero, nearn, use_weight_based_threshold, quant_thresholds, penalty_interval, thresholds, score_vals):
        super().__init__()
        self.strides = strides
        self.exclude_zero = exclude_zero
        self.nearn = nearn
        self.use_weight_based_threshold = use_weight_based_threshold
        self.quant_thresholds = quant_thresholds
        self.penalty_interval = penalty_interval
        self.thresholds = thresholds
        self.score_vals = score_vals
        self.weights = np.array([2**i for i in range(9)])[::-1]
        self.score_quantization_level = len(quant_thresholds) + 1

########################### Level 3 classes: JustConv2DLayer, ... ###########################
# Under ConvLayer class
class JustConv2DLayer(ConvLayer):
    def __init__(self, kernel_size, strides, input_max, filters):
        super().__init__(kernel_size, strides, input_max)
        self.filters = filters

class MatchJustConv2DLayer(ConvLayer):
    def __init__(self, kernel_size, strides, conv_thresholds, input_max, filters, scale):
        super().__init__(kernel_size, strides, input_max)
        self.conv_thresholds = conv_thresholds
        self.filters = filters
        self.scale = scale

class MatchConv2DLayer(ConvLayer):
    def __init__(self, kernel_size, strides, conv_thresholds, input_max, filters):
        super().__init__(kernel_size, strides, input_max)
        self.conv_thresholds = conv_thresholds
        self.filters = filters

class MatchConv2DLayer_old(ConvLayer):
    def __init__(self, kernel_size, strides, conv_thresholds, input_max, filters):
        super().__init__(kernel_size, strides, input_max)
        self.conv_thresholds = conv_thresholds
        self.filters = filters
            
class n_MatchConv2DLayer(ConvLayer):
    def __init__(self, kernel_size, strides, conv_thresholds, input_max, filters, filter_num_for_conv):
        super().__init__(kernel_size, strides, input_max)
        self.conv_thresholds = conv_thresholds
        self.filters = filters
        self.filter_num_for_conv = filter_num_for_conv
        if (len(filters.shape)!=4):
            print("n_Conv2DLayer() generates a single convolutional image using multiple filters. Please input a 4-dimensional array for the filters.")
            print("The first dimension must represent the number of cases in which the image is divided,")
            print("the second dimension must represent the number of filters used in a single convolutional image,")
            print("and the third and fourth dimensions represent the 2D filters.")
            raise Exception("filters shape error")
        
class n_MatchConv2DLayer_old(ConvLayer):
    def __init__(self, kernel_size, strides, conv_thresholds, input_max, filters, filter_num_for_conv):
        super().__init__(kernel_size, strides, input_max)
        self.conv_thresholds = conv_thresholds
        self.filters = filters
        self.filter_num_for_conv = filter_num_for_conv
        if (len(filters.shape)!=4):
            print("n_Conv2DLayer() generates a single convolutional image using multiple filters. Please input a 4-dimensional array for the filters.")
            print("The first dimension must represent the number of cases in which the image is divided,")
            print("the second dimension must represent the number of filters used in a single convolutional image,")
            print("and the third and fourth dimensions represent the 2D filters.")
            raise Exception("filters shape error")

# Under Top_ConvLayer class
class Top_JustConv2DLayer(Top_ConvLayer):
    def __init__(self, label_num, kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order):
        super().__init__(kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order)
        self.filters = np.empty(shape=(label_num,), dtype=object)
        
class Top_MatchJustConv2DLayer(Top_ConvLayer):
    def __init__(self, label_num, kernel_size, strides, use_weight_based_threshold, conv_thresholds, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, scale, dynamic_kernel_order):
        super().__init__(kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order)
        self.conv_thresholds = conv_thresholds
        self.scale = scale
        self.filters = np.empty(shape=(label_num,), dtype=object)

class Top_MatchConv2DLayer(Top_ConvLayer):
    def __init__(self, label_num, kernel_size, strides, use_weight_based_threshold, conv_thresholds, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order):
        super().__init__(kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order)
        self.conv_thresholds = conv_thresholds
        self.filters = np.empty(shape=(label_num,), dtype=object)

class Top_MatchConv2DLayer_old(Top_ConvLayer):
    def __init__(self, label_num, kernel_size, strides, use_weight_based_threshold, conv_thresholds, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order):
        super().__init__(kernel_size, strides, use_weight_based_threshold, input_max, all_ctcam_creation, top_num, top_thresholds, filter_weight_min, dynamic_kernel_order)
        self.conv_thresholds = conv_thresholds
        self.filters = np.empty(shape=(label_num,), dtype=object)