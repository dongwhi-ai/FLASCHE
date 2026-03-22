import numpy as np
import itertools
from tqdm import tqdm

from src.cam.cam import CTCAM
from src.filter.flasche_filter import FLASCHEFilter
from src.layer.flasche_layer import *
from src.utils.io import *

########################### Model class ###########################
class FLASCHE():

    ################# Model initialization ####################
    def __init__(self, label_num, input_shape):
        self.label_num = label_num
        self.input_shape = input_shape
        self.layers = []
        self.dims = []
        self.ctcams = [[] for _ in range(label_num)]
        self.data_size = [input_shape]
        self.dummy_filters = [FLASCHEFilter() for _ in range(label_num)]
        self.dummy_ctcams = [CTCAM(length=0, kernel_size=(0, 0)) for _ in range(label_num)]
        self.ctcams_sorted = [[] for _ in range(label_num)]
        self.filter_weights = [[] for _ in range(label_num)]
        self.use_weight_based_threshold = False

    def calc_output_data_size(self, input_size, kernel_size, strides):
        y_quotient, y_remainder = divmod((input_size[0]-kernel_size[0]), strides[0])
        y_temp = y_quotient + 1
        x_quotient, x_remainder = divmod((input_size[1]-kernel_size[1]), strides[1])
        x_temp = x_quotient + 1
        if ((x_remainder==0) and (y_remainder==0)):
            return (y_temp, x_temp)
        else:
            raise Exception("Not a valid structure")
        
    def compute_weight_sum_by_filter(self, filter_list):
        shape = tuple(f.shape[0] for f in filter_list)
        result_tensor = np.zeros(shape, dtype=int)

        for index_tuple in itertools.product(*[range(s) for s in shape]):
            weight_sum = sum(
                filter_list[i][idx].sum()
                for i, idx in enumerate(index_tuple)
            )
            result_tensor[index_tuple] = weight_sum

        return result_tensor.insert(0, 1)

    def compute_weight_sum(self, label):
        weight_sum = None
        for layer_num, layer in enumerate(self.layers):
            if (issubclass(type(layer), ConvLayer)):
                weights_temp = np.sum(layer.filters, axis=(-2, -1))
                if weight_sum is None:
                    weight_sum = weights_temp
                else:
                    weight_sum = weight_sum[..., np.newaxis] * weights_temp.reshape((1,) * weight_sum.ndim + (-1,))
            elif issubclass(type(layer), Top_ConvLayer):
                weights_temp = np.sum(layer.filters[label], axis=(-2, -1))
                if weight_sum is None:
                    weight_sum = weights_temp
                else:
                    weight_sum = np.expand_dims(weight_sum, axis=-1) * weights_temp

            self.filter_weights[label].append(weight_sum)
        return None
    
    def compute_weight_sum_all(self):
        for label in range(self.label_num):
            self.compute_weight_sum(label=label)
        
        return None

    def model_validity_check(self):
        y_temp, x_temp = self.input_shape
        for layer in self.layers:
            input_size_temp = (y_temp, x_temp)
            if (issubclass(type(layer), ConvLayer) or issubclass(type(layer), Top_ConvLayer)):
                y_temp, x_temp = self.calc_output_data_size(input_size=input_size_temp, kernel_size=layer.kernel_size, strides=layer.strides)
            elif isinstance(layer, ScoringLayer):
                y_temp, x_temp = self.calc_output_data_size(input_size=input_size_temp, kernel_size=(3, 3), strides=layer.strides)
            elif isinstance(layer, MaxPooling2DLayer):
                y_temp, x_temp = self.calc_output_data_size(input_size=input_size_temp, kernel_size=layer.pool_size, strides=layer.strides)
            elif isinstance(layer, ZeroPaddingLayer):
                y_temp = y_temp + np.sum(layer.pad_width[0])
                x_temp = x_temp + np.sum(layer.pad_width[1])
            elif isinstance(layer, NormalizationLayer):
                pass
            else:
                pass
        return (y_temp, x_temp)

    def model_structure_info(self):
        y_temp, x_temp = self.input_shape
        for i, layer in enumerate(self.layers):
            print(f"{i} layer input shape: ({y_temp}, {x_temp})      {layer.__class__.__name__}")
            y_temp, x_temp = self.data_size[i+1]
        print(f"{i} layer output shape: ({y_temp}, {x_temp})")
        print("valid structure!")
        return None

    def add(self, layer):
        shape = np.array(self.dims)
        shape = shape[shape != 0]  
        
        self.layers.append(layer)
        y_temp, x_temp = self.model_validity_check()
        self.data_size.append((y_temp, x_temp))

        if (issubclass(type(layer), ConvLayer)):
            self.dims.append(len(layer.filters))
            for i in range(self.label_num):
                self.ctcams[i].append(None)

        elif (issubclass(type(layer), Top_ConvLayer)):
            for i in range(self.label_num):
                ctcam_array = np.empty(shape=shape, dtype=object)
                for index in np.ndindex(*shape):
                    if (layer.all_ctcam_creation):
                        ctcam_array[index] = CTCAM(length=2**layer.bits, kernel_size=layer.kernel_size).make_ctcam()
                    else:
                        ctcam_array[index] = CTCAM(length=0, kernel_size=layer.kernel_size).make_ctcam()
                self.ctcams[i].append(ctcam_array)
            if (layer.use_weight_based_threshold):
                self.use_weight_based_threshold = True
            else:
                pass
            if (layer.dynamic_kernel_order):
                self.dims.append(layer.top_num)
                filter_array = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int) 
                for i in range(self.label_num):
                    layer.filters[i] = filter_array
            else:
                def all_binary_kernels(x: int, y: int, *, dtype=np.uint8, bit_order: str = "msb_first"):
                    n = x * y
                    if n == 0:
                        return np.zeros((1, x, y), dtype=dtype)

                    # 0..2**n-1 (정수)
                    vals = np.arange(1 << n, dtype=np.uint64) # uint64

                    # 시프트 양
                    shifts = np.arange(n, dtype=np.uint64)[::-1]  # MSB가 왼쪽 열

                    bits = (vals[:, None] >> shifts) & np.uint64(1)

                    if bit_order == "lsb_first":
                        bits = bits[:, ::-1]
                    elif bit_order != "msb_first":
                        raise ValueError("bit_order must be 'msb_first' or 'lsb_first'")

                    return bits.reshape(-1, x, y).astype(dtype, copy=False)
                self.dims.append((2**layer.bits)-1)
                all_kernels = all_binary_kernels(layer.kernel_size[0], layer.kernel_size[1], bit_order="msb_first")[1:]
                filter_array = np.zeros(shape=(*shape, (2**layer.bits)-1, *layer.kernel_size), dtype=int)
                for idx in np.ndindex(*shape):
                    filter_array[idx] = all_kernels
                for i in range(self.label_num):
                    layer.filters[i] = filter_array
        elif isinstance(layer, ScoringLayer):
            self.dims.append(0)
            for i in range(self.label_num):
                array = np.empty(shape=(*shape, *(y_temp, x_temp)), dtype=object)
                for index in np.ndindex(*shape, *(y_temp, x_temp)):
                    array[index] = CTCAM(length=2**9, kernel_size=(3, 3)).make_ctcam()
                self.ctcams[i].append(array)
            if (layer.use_weight_based_threshold):
                self.use_weight_based_threshold = True
            else:
                pass
            self.model_compile()
    
        else:
            self.dims.append(0)
            for i in range(self.label_num):
                self.ctcams[i].append(None)

    def model_compile(self):
        self.model_structure_info()
        self.compute_weight_sum_all()
        print("\n--- Compile done ---")

    def process_JustConv2DLayer(self, label, shape, image, layer_num, layer=JustConv2DLayer):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        filtered_images_arr = np.empty(shape=(*output_shape, *self.data_size[layer_num+1]))       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].just_convolution(input_image=image[index[:-1]], 
                                                                                    kernel=layer.filters[index[-1]], 
                                                                                    strides=layer.strides, 
                                                                                    kernel_size=layer.kernel_size, 
                                                                                    output_size=self.data_size[layer_num+1])
        return filtered_images_arr
    
    def process_MatchJustConv2DLayer(self, label, shape, image, layer_num, layer=MatchJustConv2DLayer):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[index[-1]], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_just_convolution(input_image=image[index[:-1]], 
                                                                                          matched_image=matched_images_arr[index], 
                                                                                          kernel=layer.filters[index[-1]], 
                                                                                          strides=layer.strides, 
                                                                                          scale=layer.scale, 
                                                                                          kernel_size=layer.kernel_size, 
                                                                                          output_size=self.data_size[layer_num+1])
        return filtered_images_arr
    
    def process_MatchConv2DLayer(self, label, shape, image, layer_num, layer=MatchConv2DLayer):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[index[-1]], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)    
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[index[-1]], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        return filtered_images_arr
    
    def process_MatchConv2DLayer_old(self, label, shape, image, layer_num, layer=MatchConv2DLayer_old):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri_together(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[index[-1]], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)    
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[index[-1]], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        return filtered_images_arr

    def process_n_MatchConv2DLayer(self, label, shape, image, layer_num, layer=n_MatchConv2DLayer):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, layer.filter_num_for_conv, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].n_filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                       kernels=layer.filters[index[-1]], 
                                                                                       strides=layer.strides, 
                                                                                       kernel_size=layer.kernel_size, 
                                                                                       output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)    
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].n_match_convolution(input_image=image[index[:-1]], 
                                                                                       matched_images=matched_images_arr[index], 
                                                                                       kernels=layer.filters[index[-1]], 
                                                                                       strides=layer.strides, 
                                                                                       kernel_size=layer.kernel_size, 
                                                                                       output_size=self.data_size[layer_num+1])
        return filtered_images_arr
    
    def process_n_MatchConv2DLayer_old(self, label, shape, image, layer_num, layer=n_MatchConv2DLayer_old):
        last_dim = len(layer.filters)
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, layer.filter_num_for_conv, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].n_filter_matcher_tri_together(quantized_image=quantized_images[index[:-1]], 
                                                                                       kernels=layer.filters[index[-1]], 
                                                                                       strides=layer.strides, 
                                                                                       kernel_size=layer.kernel_size, 
                                                                                       output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)    
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].n_match_convolution(input_image=image[index[:-1]], 
                                                                                       matched_images=matched_images_arr[index], 
                                                                                       kernels=layer.filters[index[-1]], 
                                                                                       strides=layer.strides, 
                                                                                       kernel_size=layer.kernel_size, 
                                                                                       output_size=self.data_size[layer_num+1])
        return filtered_images_arr

    def process_Top_JustConv2DLayer(self, label, shape, image, layer_num, layer=Top_JustConv2DLayer):
        if (layer.dynamic_kernel_order):
            last_dim = layer.top_num
            output_shape = tuple(shape) + (last_dim,)
            
            layer.filters[label] = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int)
            quantized_images_for_top = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)

            if (layer.top_quantization_level==2):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        bi_threshold = layer.top_thresholds[0] * self.filter_weights[label][layer_num][index]
                    else:
                        bi_threshold = layer.top_thresholds[0]
                    quantized_images_for_top[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                                threshold=bi_threshold)
                    patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_top[index], 
                                                                            pattern_size=layer.kernel_size, 
                                                                            strides=(1, 1),
                                                                            weights=layer.weights, 
                                                                            output_size=self.data_size[layer_num+1])
                    
                    if (layer.all_ctcam_creation):
                        self.dummy_ctcams[label].patterns_counting(patterns=patterns, 
                                                                ctcam=self.ctcams[label][layer_num][index])
                        ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                        layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                            kernel_size=layer.kernel_size, 
                                                                                            top_num=layer.top_num, 
                                                                                            filter_weight_min=layer.filter_weight_min)
                    else:
                        self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns, 
                                                                                                                        ctcam=self.ctcams[label][layer_num][index])
                        ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                        layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                            kernel_size=layer.kernel_size, 
                                                                                                            top_num=layer.top_num, 
                                                                                                            filter_weight_min=layer.filter_weight_min)
                    if (layer.use_weight_based_threshold):
                        self.filters[layer_num] = layer.use_weight_based_threshold
                    
            elif (layer.top_quantization_level==3):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        tri_threshold1 = layer.top_thresholds[0] * self.filter_weights[label][layer_num][index]
                        tri_threshold2 = layer.top_thresholds[1] * self.filter_weights[label][layer_num][index]
                    else:
                        tri_threshold1 = layer.top_thresholds[0]
                        tri_threshold2 = layer.top_thresholds[1]
                    quantized_images_for_top[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                                threshold1=tri_threshold1, 
                                                                                                threshold2=tri_threshold2)
                    patterns_arr = self.dummy_ctcams[label].pattern_detection_tri(quantized_image=quantized_images_for_top[index], 
                                                                                pattern_size=layer.kernel_size, 
                                                                                strides=(1, 1),
                                                                                weights=layer.weights, 
                                                                                output_size=self.data_size[layer_num+1])

                    if (layer.all_ctcam_creation):
                        self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr, 
                                                                ctcam=self.ctcams[label][layer_num][index])
                        ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                        layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                            kernel_size=layer.kernel_size, 
                                                                                            top_num=layer.top_num, 
                                                                                            filter_weight_min=layer.filter_weight_min)
                    else:
                        self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns_arr, 
                                                                                                                        ctcam=self.ctcams[label][layer_num][index])
                        ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                        layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                            kernel_size=layer.kernel_size, 
                                                                                                            top_num=layer.top_num, 
                                                                                                            filter_weight_min=layer.filter_weight_min)

            
            filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
            for index in np.ndindex(*output_shape):
                filtered_images_arr[index] = self.dummy_filters[label].just_convolution(input_image=image[index[:-1]], 
                                                                                        kernel=layer.filters[label][index], 
                                                                                        strides=layer.strides, 
                                                                                        kernel_size=layer.kernel_size, 
                                                                                        output_size=self.data_size[layer_num+1])
            
            if (layer.use_weight_based_threshold):
                self.compute_weight_sum(label=label)
        else:
            last_dim = (2**layer.bits)-1
            output_shape = tuple(shape) + (last_dim,)

            quantized_images_for_top = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)

            for index in np.ndindex(*shape):
                bi_threshold = layer.top_thresholds[0]
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                            threshold=bi_threshold)
                patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_top[index], 
                                                                        pattern_size=layer.kernel_size, 
                                                                        strides=(1, 1),
                                                                        weights=layer.weights, 
                                                                        output_size=self.data_size[layer_num+1])
                

                self.dummy_ctcams[label].patterns_counting(patterns=patterns, 
                                                        ctcam=self.ctcams[label][layer_num][index])
            filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
            for index in np.ndindex(*output_shape):
                filtered_images_arr[index] = self.dummy_filters[label].just_convolution(input_image=image[index[:-1]], 
                                                                                        kernel=layer.filters[label][index], 
                                                                                        strides=layer.strides, 
                                                                                        kernel_size=layer.kernel_size, 
                                                                                        output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr
    
    def process_static_kernel_CTCAM_reorder(self, shape, layer_num, layer, scoringlayer_idx):
        for label in range(self.label_num):
            layer.filters[label] = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int)
            #print('here')
            #print(label, layer_num)
            #print(self.ctcams[label][scoringlayer_idx])
            #print(self.ctcams[label][scoringlayer_idx].shape)
            score_ctcams_reordered = np.zeros(shape=(layer.top_num, *(self.ctcams[label][scoringlayer_idx].shape[1:])), dtype=object)
            for index in np.ndindex(*shape):
                if (layer.all_ctcam_creation):
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                        kernel_size=layer.kernel_size, 
                                                                                        top_num=layer.top_num, 
                                                                                        filter_weight_min=layer.filter_weight_min)
                    flat = layer.filters[label][index].reshape(layer.top_num, -1).astype(np.uint8)
                    vals = (flat.astype(object) * layer.weights).sum(axis=1)
                    vals = vals - 1
                    vals_idx = np.array([int(v) for v in vals], dtype=np.intp)
                    score_ctcams_reordered[index] = self.ctcams[label][scoringlayer_idx][index][vals_idx]
                    
                else:
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
                    flat = layer.filters[label][index].reshape(layer.top_num, -1).astype(np.uint8)
                    vals = (flat.astype(object) * layer.weights).sum(axis=1)
                    vals = vals - 1
                    vals_idx = np.array([int(v) for v in vals], dtype=np.intp)
                    score_ctcams_reordered[index] = self.ctcams[label][scoringlayer_idx][index][vals_idx]
            self.ctcams[label][scoringlayer_idx] = score_ctcams_reordered
            
            if (layer.use_weight_based_threshold):
                self.compute_weight_sum(label=label)
        
        return None

    def process_Top_JustConv2DLayer_predict(self, label, shape, image, layer_num, layer=Top_JustConv2DLayer):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)

        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].just_convolution(input_image=image[index[:-1]], 
                                                                                    kernel=layer.filters[label][index], 
                                                                                    strides=layer.strides, 
                                                                                    kernel_size=layer.kernel_size, 
                                                                                    output_size=self.data_size[layer_num+1])
        return filtered_images_arr

    def process_Top_MatchJustConv2DLayer(self, label, shape, image, layer_num, layer=Top_MatchJustConv2DLayer):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)
        
        layer.filters[label] = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int)
        quantized_images_for_top = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)

        if (layer.top_quantization_level==2):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                            threshold=layer.top_thresholds[0])
                patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_top[index], 
                                                                         pattern_size=layer.kernel_size, 
                                                                         strides=(1, 1),
                                                                         weights=layer.weights, 
                                                                         output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        elif (layer.top_quantization_level==3):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                             threshold1=layer.top_thresholds[0], 
                                                                                             threshold2=layer.top_thresholds[1])
                patterns_arr = self.dummy_ctcams[label].pattern_detection_tri(quantized_image=quantized_images_for_top[index], 
                                                                              pattern_size=layer.kernel_size, 
                                                                              strides=(1, 1),
                                                                              weights=layer.weights, 
                                                                              output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns_arr, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        
        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_just_convolution(input_image=image[index[:-1]], 
                                                                                          matched_image=matched_images_arr[index], 
                                                                                          kernel=layer.filters[label][index], 
                                                                                          strides=layer.strides, 
                                                                                          scale=layer.scale, 
                                                                                          kernel_size=layer.kernel_size, 
                                                                                          output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr



    def process_Top_MatchJustConv2DLayer_predict(self, label, shape, image, layer_num, layer=Top_MatchJustConv2DLayer):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_just_convolution(input_image=image[index[:-1]], 
                                                                                          matched_image=matched_images_arr[index], 
                                                                                          kernel=layer.filters[label][index], 
                                                                                          strides=layer.strides, 
                                                                                          scale=layer.scale, 
                                                                                          kernel_size=layer.kernel_size, 
                                                                                          output_size=self.data_size[layer_num+1])
        return filtered_images_arr

    def process_Top_MatchConv2DLayer(self, label, shape, image, layer_num, layer=Top_MatchConv2DLayer):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)
        
        layer.filters[label] = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int)
        quantized_images_for_top = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)

        if (layer.top_quantization_level==2):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                            threshold=layer.top_thresholds[0])
                patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_top[index], 
                                                                         pattern_size=layer.kernel_size, 
                                                                         strides=(1, 1),
                                                                         weights=layer.weights, 
                                                                         output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        elif (layer.top_quantization_level==3):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                             threshold1=layer.top_thresholds[0], 
                                                                                             threshold2=layer.top_thresholds[1])
                patterns_arr = self.dummy_ctcams[label].pattern_detection_tri(quantized_image=quantized_images_for_top[index], 
                                                                              pattern_size=layer.kernel_size, 
                                                                              strides=(1, 1),
                                                                              weights=layer.weights, 
                                                                              output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns_arr, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        
        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr
    
    def process_Top_MatchConv2DLayer_old(self, label, shape, image, layer_num, layer=Top_MatchConv2DLayer_old):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)
        
        layer.filters[label] = np.zeros(shape=(*shape, layer.top_num, *layer.kernel_size), dtype=int)
        quantized_images_for_top = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)

        if (layer.top_quantization_level==2):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                            threshold=layer.top_thresholds[0])
                patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_top[index], 
                                                                         pattern_size=layer.kernel_size, 
                                                                         strides=(1, 1),
                                                                         weights=layer.weights, 
                                                                         output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        elif (layer.top_quantization_level==3):
            for index in np.ndindex(*shape):
                quantized_images_for_top[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                             threshold1=layer.top_thresholds[0], 
                                                                                             threshold2=layer.top_thresholds[1])
                patterns_arr = self.dummy_ctcams[label].pattern_detection_tri(quantized_image=quantized_images_for_top[index], 
                                                                              pattern_size=layer.kernel_size, 
                                                                              strides=(1, 1),
                                                                              weights=layer.weights, 
                                                                              output_size=self.data_size[layer_num+1])
                if (layer.all_ctcam_creation):
                    self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr, 
                                                               ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters(ctcam=ctcam_sorted_temp, 
                                                                                         kernel_size=layer.kernel_size, 
                                                                                         top_num=layer.top_num, 
                                                                                         filter_weight_min=layer.filter_weight_min)
                else:
                    self.ctcams[label][layer_num][index] = self.dummy_ctcams[label].patterns_counting_ctcam_creation(patterns=patterns_arr, 
                                                                                                                     ctcam=self.ctcams[label][layer_num][index])
                    ctcam_sorted_temp = self.dummy_ctcams[label].ctcam_sorting(ctcam=self.ctcams[label][layer_num][index])
                    layer.filters[label][index] = self.dummy_ctcams[label].top_n_filters_ctcam_creation(ctcam=ctcam_sorted_temp, 
                                                                                                        kernel_size=layer.kernel_size, 
                                                                                                        top_num=layer.top_num, 
                                                                                                        filter_weight_min=layer.filter_weight_min)
        
        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri_together(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr

    def process_Top_MatchConv2DLayer_predict(self, label, shape, image, layer_num, layer=Top_MatchConv2DLayer):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr
    
    def process_Top_MatchConv2DLayer_old_predict(self, label, shape, image, layer_num, layer=Top_MatchConv2DLayer_old):
        last_dim = layer.top_num
        output_shape = tuple(shape) + (last_dim,)

        quantized_images = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        for index in np.ndindex(*shape):
            quantized_images[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                 threshold1=layer.conv_thresholds[0], 
                                                                                 threshold2=layer.conv_thresholds[1])
        matched_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*output_shape):
            matched_images_arr[index] = self.dummy_filters[label].filter_matcher_tri_together(quantized_image=quantized_images[index[:-1]], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
        filtered_images_arr = np.zeros(shape=(*output_shape, *self.data_size[layer_num+1]), dtype=int)       
        for index in np.ndindex(*output_shape):
            filtered_images_arr[index] = self.dummy_filters[label].match_convolution(input_image=image[index[:-1]], 
                                                                                     matched_image=matched_images_arr[index], 
                                                                                     kernel=layer.filters[label][index], 
                                                                                     strides=layer.strides, 
                                                                                     kernel_size=layer.kernel_size, 
                                                                                     output_size=self.data_size[layer_num+1])
                
        return filtered_images_arr


    def process_ZeroPaddingLayer(self, label, shape, image, layer_num, layer=ZeroPaddingLayer):
        padded_images = self.dummy_filters[label].zero_padding(input_image=image, 
                                                               pad_width=layer.pad_width)
        return padded_images

    def process_MaxPooling2DLayer(self, label, shape, image, layer_num, layer=MaxPooling2DLayer):
        pooled_images = np.zeros(shape=(*shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*shape):
            pooled_images[index] = self.dummy_filters[label].max_pooling(input_image=image[index], 
                                                                         strides=layer.strides, 
                                                                         pool_size=layer.pool_size, 
                                                                         output_size=self.data_size[layer_num+1])
        return pooled_images

    def process_NormalizationLayer(self, label, shape, image, layer_num, layer=NormalizationLayer):
        if (layer.normalize_per_image):
            normalized_image = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
            if (layer.adaptability=='Weak'):
                prelayer = self.layers[layer_num-1]
                if (isinstance(prelayer, JustConv2DLayer)):
                    prelayer_filter_weight_sum = np.zeros(shape=(len(prelayer.filters),), dtype=int)
                    for filter_num in range(len(prelayer.filters)):
                        prelayer_filter_weight_sum[filter_num] = np.sum(prelayer.filters[filter_num])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=0, 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index[-1]])
                elif (isinstance(prelayer, MatchJustConv2DLayer)):
                    prelayer_filter_weight_sum = np.zeros(shape=(len(prelayer.filters),), dtype=int)
                    for filter_num in range(len(prelayer.filters)):
                        prelayer_filter_weight_sum[filter_num] = np.sum(prelayer.filters[filter_num])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=0, 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index[-1]]*prelayer.scale)
                elif (isinstance(prelayer, MatchConv2DLayer)):
                    prelayer_filter_weight_sum = np.zeros(shape=(len(prelayer.filters),), dtype=int)
                    for filter_num in range(len(prelayer.filters)):
                        prelayer_filter_weight_sum[filter_num] = np.sum(prelayer.filters[filter_num])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=prelayer.conv_thresholds[1]*prelayer_filter_weight_sum[index[-1]], 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index[-1]])
                elif (isinstance(prelayer, n_MatchConv2DLayer) or isinstance(prelayer, n_MatchConv2DLayer_old)):
                    prelayer_filter_weight_sum = np.zeros(shape=(len(prelayer.filters), prelayer.filter_num_for_conv), dtype=int)
                    for filter_num_row in range(len(prelayer.filters)):
                        for filter_num_col in range(prelayer.filter_num_for_conv):
                            prelayer_filter_weight_sum[filter_num_row][filter_num_col] = np.sum(prelayer.filters[filter_num_row][filter_num_col])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=prelayer.conv_thresholds[1]*np.min(prelayer_filter_weight_sum[index[-1]]), 
                                                                                      old_max=prelayer.input_max*np.max(prelayer_filter_weight_sum[index[-1]]))
                elif (isinstance(prelayer, Top_JustConv2DLayer)):
                    prelayer_filter_weight_sum = np.zeros(shape=shape, dtype=int)
                    for index in np.ndindex(*shape):
                        prelayer_filter_weight_sum[index] = np.sum(prelayer.filters[label][index])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=0, 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index])
                elif (isinstance(prelayer, Top_MatchJustConv2DLayer)):
                    prelayer_filter_weight_sum = np.zeros(shape=shape, dtype=int)
                    for index in np.ndindex(*shape):
                        prelayer_filter_weight_sum[index] = np.sum(prelayer.filters[label][index])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching,  
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=0, 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index]*prelayer.scale)
                elif (isinstance(prelayer, Top_MatchConv2DLayer) or isinstance(prelayer, Top_MatchConv2DLayer_old)):
                    prelayer_filter_weight_sum = np.zeros(shape=shape, dtype=int)
                    for index in np.ndindex(*shape):
                        prelayer_filter_weight_sum[index] = np.sum(prelayer.filters[label][index])
                    for index in np.ndindex(*shape):
                        normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                      stretching=layer.stretching, 
                                                                                      new_max=layer.new_max, 
                                                                                      old_min=prelayer.conv_thresholds[1]*prelayer_filter_weight_sum[index], 
                                                                                      old_max=prelayer.input_max*prelayer_filter_weight_sum[index])
            elif (layer.adaptability=='Strong'):
                for index in np.ndindex(*shape):
                    normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                                   stretching=layer.stretching, 
                                                                                   new_max=layer.new_max, 
                                                                                   old_min=np.min(image[index][image[index] > 0]) if np.any(image[index] > 0) else 0, 
                                                                                   old_max=np.max(image[index]))

            else:
                pass
        elif (layer.normalize_per_group):
            normalized_image = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
            for index in np.ndindex(*shape[:-1]):
                normalized_image[index] = self.dummy_filters[label].normalize(image=image[index], 
                                                                              stretching=layer.stretching, 
                                                                              new_max=layer.new_max, 
                                                                              old_min=layer.old_min, 
                                                                              old_max=layer.old_max)
        else:
            normalized_image = self.dummy_filters[label].normalize(image=image, 
                                                                   stretching=layer.stretching, 
                                                                   new_max=layer.new_max, 
                                                                   old_min=layer.old_min, 
                                                                   old_max=layer.old_max)

        return normalized_image
    
    def process_ScoringLayer(self, label, shape, image, layer_num, layer=ScoringLayer):
        quantized_images_for_score = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        if (layer.nearn):
            if (layer.score_quantization_level==2):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        bi_threshold = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
                    else:
                        bi_threshold = layer.quant_thresholds[0]
                    quantized_images_for_score[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                                threshold=bi_threshold)
                    patterns_arr = self.dummy_ctcams[label].pattern_detection_nearn_bi(quantized_image=quantized_images_for_score[index], 
                                                                                    pattern_size=(3, 3), 
                                                                                    strides=layer.strides, 
                                                                                    weights=layer.weights, 
                                                                                    output_size=self.data_size[layer_num+1])
                    for index_score in np.ndindex(*self.data_size[layer_num+1]):
                        self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr[index_score], 
                                                                ctcam=self.ctcams[label][layer_num][index][index_score])
            elif (layer.score_quantization_level==3):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        tri_threshold1 = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
                        tri_threshold2 = layer.quant_thresholds[1] * self.filter_weights[label][layer_num][index]
                    else:
                        tri_threshold1 = layer.quant_thresholds[0]
                        tri_threshold2 = layer.quant_thresholds[1]
                    quantized_images_for_score[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                                threshold1=tri_threshold1, 
                                                                                                threshold2=tri_threshold2)
                    patterns_arr = self.dummy_ctcams[label].pattern_detection_nearn_tri(quantized_image=quantized_images_for_score[index], 
                                                                                    pattern_size=(3, 3), 
                                                                                    strides=layer.strides, 
                                                                                    weights=layer.weights, 
                                                                                    output_size=self.data_size[layer_num+1])
                    for index_score in np.ndindex(*self.data_size[layer_num+1]):
                        self.dummy_ctcams[label].patterns_counting(patterns=patterns_arr[index_score], 
                                                                ctcam=self.ctcams[label][layer_num][index][index_score])

        else:
            if (layer.score_quantization_level==2):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        bi_threshold = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
                    else:
                        bi_threshold = layer.quant_thresholds[0]
                    quantized_images_for_score[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                                threshold=bi_threshold)
                    patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_score[index], 
                                                                            pattern_size=(3, 3), 
                                                                            strides=layer.strides, 
                                                                            weights=layer.weights, 
                                                                            output_size=self.data_size[layer_num+1])
                    for index_score in np.ndindex(*self.data_size[layer_num+1]):
                        self.dummy_ctcams[label].pattern_counting(pattern=patterns[index_score], 
                                                                ctcam=self.ctcams[label][layer_num][index][index_score])
            elif (layer.score_quantization_level==3):
                for index in np.ndindex(*shape):
                    if (layer.use_weight_based_threshold):
                        tri_threshold1 = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
                        tri_threshold2 = layer.quant_thresholds[1] * self.filter_weights[label][layer_num][index]
                    else:
                        tri_threshold1 = layer.quant_thresholds[0]
                        tri_threshold2 = layer.quant_thresholds[1]
                    quantized_images_for_score[index] = self.dummy_filters[label].quantization_tri(input_image=image[index], 
                                                                                                threshold1=tri_threshold1, 
                                                                                                threshold2=tri_threshold2)
                    patterns_arr = self.dummy_ctcams[label].pattern_detection_tri(quantized_image=quantized_images_for_score[index], 
                                                                            pattern_size=(3, 3), 
                                                                            strides=layer.strides, 
                                                                            weights=layer.weights, 
                                                                            output_size=self.data_size[layer_num+1])
                    for index_score in np.ndindex(*self.data_size[layer_num+1]):
                        self.dummy_ctcams[label].pattern_counting(pattern=patterns_arr[index_score], 
                                                                ctcam=self.ctcams[label][layer_num][index][index_score])
        return None

    def process_ScoringLayer_intermediate(self, label, shape, image, layer_num, layer=ScoringLayer):
        quantized_images_for_score = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        patterns_arr = np.zeros(shape=(*shape, *self.data_size[layer_num+1]), dtype=int)
        for index in np.ndindex(*shape):
            if (layer.use_weight_based_threshold):
                bi_threshold = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
            else:
                bi_threshold = layer.quant_thresholds[0]
            quantized_images_for_score[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                          threshold=bi_threshold)
            patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_score[index], 
                                                                     pattern_size=(3, 3), 
                                                                     strides=layer.strides, 
                                                                     weights=layer.weights, 
                                                                     output_size=self.data_size[layer_num+1])
            patterns_arr[index] = patterns
        return patterns_arr

    def process_ScoringLayer_finalize(self, label, shape, patterns_arr, layer=ScoringLayer):
        score = 0
        label_scores_each = np.zeros(shape=shape, dtype=int)
        every_scores = np.zeros(shape=shape, dtype=object)
        for index in np.ndindex(*shape):
            score_temp, score_detailed = self.dummy_ctcams[label].calc_score(ctcams_sorted=self.ctcams_sorted[label][index], 
                                                             patterns=patterns_arr[index], 
                                                             exclude_zero=layer.exclude_zero, 
                                                             penalty_interval=layer.penalty_interval, 
                                                             thresholds=layer.thresholds, 
                                                             score_vals=layer.score_vals)
            score += score_temp
            label_scores_each[index] = score_temp
            every_scores[index] = score_detailed
        return int(score), label_scores_each, every_scores

    def process_ScoringLayer_predict(self, label, shape, image, layer_num, layer=ScoringLayer):
        quantized_images_for_score = np.zeros(shape=(*shape, *self.data_size[layer_num]), dtype=int)
        score = 0
        label_scores_each = np.zeros(shape=shape, dtype=int)
        every_scores = np.zeros(shape=shape, dtype=object)
        for index in np.ndindex(*shape):
            if (layer.use_weight_based_threshold):
                bi_threshold = layer.quant_thresholds[0] * self.filter_weights[label][layer_num][index]
            else:
                bi_threshold = layer.quant_thresholds[0]
            quantized_images_for_score[index] = self.dummy_filters[label].quantization_bi(input_image=image[index], 
                                                                                          threshold=bi_threshold)
            patterns = self.dummy_ctcams[label].pattern_detection_bi(quantized_image=quantized_images_for_score[index], 
                                                                     pattern_size=(3, 3), 
                                                                     strides=layer.strides, 
                                                                     weights=layer.weights, 
                                                                     output_size=self.data_size[layer_num+1])
            score_temp, score_detailed = self.dummy_ctcams[label].calc_score(ctcams_sorted=self.ctcams_sorted[label][index], 
                                                             patterns=patterns, 
                                                             exclude_zero=layer.exclude_zero, 
                                                             penalty_interval=layer.penalty_interval, 
                                                             thresholds=layer.thresholds, 
                                                             score_vals=layer.score_vals)
            score += score_temp
            label_scores_each[index] = score_temp
            every_scores[index] = score_detailed
        return int(score), label_scores_each, every_scores



    def fit(self, x, y, save_options):
        print("\nModel training started...")

        for image_num in tqdm(range(len(x))):
            image_save = False
            image = x[image_num]
            label = y[image_num]
            if (image_num in save_options["save_image_indices"]):
                image_save = True
            if (image_save):
                image_name = "train_"+str(image_num)+"image"
                save_image(images=image, save_dir=save_options["save_image_dir"], image_name=image_name, image_type=save_options["save_image_type"], one_file=save_options["save_image_one_file"], float_type=save_options["save_image_float"], use_hex=save_options["save_image_hex"])
            
            for layer_num, layer in enumerate(self.layers):

                shape = np.array(self.dims[:layer_num], dtype=int)
                shape = shape[shape != 0]  
                
                # JustConv2DLayer
                if isinstance(layer, JustConv2DLayer):
                    image = self.process_JustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # MatchJustConv2DLayer
                elif isinstance(layer, MatchJustConv2DLayer):
                    image = self.process_MatchJustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Conv2DLayer
                elif isinstance(layer, MatchConv2DLayer):
                    image = self.process_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Conv2DLayer_old
                elif isinstance(layer, MatchConv2DLayer_old):
                    image = self.process_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # n_Conv2DLayer
                elif isinstance(layer, n_MatchConv2DLayer):
                    image = self.process_n_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # n_Conv2DLayer_old
                elif isinstance(layer, n_MatchConv2DLayer_old):
                    image = self.process_n_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Top_JustConv2DLayer
                elif isinstance(layer, Top_JustConv2DLayer):
                    image = self.process_Top_JustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Top_MatchJustConv2DLayer
                elif isinstance(layer, Top_MatchJustConv2DLayer):
                    image = self.process_Top_MatchJustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Top_Conv2DLayer
                elif isinstance(layer, Top_MatchConv2DLayer):
                    image = self.process_Top_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # Top_Conv2DLayer_old
                elif isinstance(layer, Top_MatchConv2DLayer_old):
                    image = self.process_Top_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # ZeroPaddingLayer
                elif isinstance(layer, ZeroPaddingLayer):
                    image = self.process_ZeroPaddingLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # MaxPooling2DLayer
                elif isinstance(layer, MaxPooling2DLayer):
                    image = self.process_MaxPooling2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # NormalizationLayer
                elif isinstance(layer, NormalizationLayer):
                    image = self.process_NormalizationLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                # ScoringLayer
                elif isinstance(layer, ScoringLayer):
                    self.process_ScoringLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                else:
                    pass
                if (image_save):
                    image_name = "train_"+str(image_num)+"image_"+str(layer_num)+"layer"
                    save_image(images=image, save_dir=save_options["save_image_dir"], image_name=image_name, image_type=save_options["save_image_type"], one_file=save_options["save_image_one_file"], float_type=save_options["save_image_float"], use_hex=save_options["save_image_hex"])
            
        self.compute_weight_sum_all()
        for layer_num, layer in enumerate(self.layers):
            if issubclass(type(layer), Top_ConvLayer):
                if (layer.dynamic_kernel_order==False):
                    self.dims[layer_num] = layer.top_num
                    shape = np.array(self.dims[:layer_num], dtype=int)
                    shape = shape[shape != 0]  
                    self.process_static_kernel_CTCAM_reorder(shape=shape, layer_num=layer_num, layer=layer, scoringlayer_idx=len(self.layers)-1)
    
    def get_intermediate_output(self, x):
        patterns_arr_set = []
        for label in range(self.label_num):
            scorectcam_shape = self.ctcams[label][-1].shape
            self.ctcams_sorted[label] = np.zeros(shape=scorectcam_shape, dtype=object)
            for index in np.ndindex(scorectcam_shape):
                self.ctcams_sorted[label][index] = self.dummy_ctcams[label].ctcam_sorting(self.ctcams[label][-1][index])
        
        for image_num in tqdm(range(len(x))):
            image = np.array(x[image_num])
            patterns_arr_temp = []
            for label in range(self.label_num):
                image = np.array(x[image_num])
                for layer_num, layer in enumerate(self.layers):
                    shape = np.array(self.dims[:layer_num], dtype=int)
                    shape = shape[shape != 0]  
                    

                    # JustConv2DLayer
                    if isinstance(layer, JustConv2DLayer):
                        image = self.process_JustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # MatchJustConv2DLayer
                    elif isinstance(layer, MatchJustConv2DLayer):
                        image = self.process_MatchJustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Conv2DLayer
                    elif isinstance(layer, MatchConv2DLayer):
                        image = self.process_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Conv2DLayer_old
                    elif isinstance(layer, MatchConv2DLayer_old):
                        image = self.process_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # n_Conv2DLayer
                    elif isinstance(layer, n_MatchConv2DLayer):
                        image = self.process_n_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # n_Conv2DLayer_old
                    elif isinstance(layer, n_MatchConv2DLayer_old):
                        image = self.process_n_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_JustConv2DLayer
                    elif isinstance(layer, Top_JustConv2DLayer):
                        image = self.process_Top_JustConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_MatchJustConv2DLayer
                    elif isinstance(layer, Top_MatchJustConv2DLayer):
                        image = self.process_Top_MatchJustConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_Conv2DLayer
                    elif isinstance(layer, Top_MatchConv2DLayer):
                        image = self.process_Top_MatchConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_Conv2DLayer_old
                    elif isinstance(layer, Top_MatchConv2DLayer_old):
                        image = self.process_Top_MatchConv2DLayer_old_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # ZeroPaddingLayer
                    elif isinstance(layer, ZeroPaddingLayer):
                        image = self.process_ZeroPaddingLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # MaxPooling2DLayer
                    elif isinstance(layer, MaxPooling2DLayer):
                        image = self.process_MaxPooling2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # NormalizationLayer
                    elif isinstance(layer, NormalizationLayer):
                        image = self.process_NormalizationLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # ScoringLayer
                    elif isinstance(layer, ScoringLayer):
                        patterns_arr = self.process_ScoringLayer_intermediate(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                patterns_arr_temp.append(patterns_arr)
            patterns_arr_set.append(patterns_arr_temp)
        return patterns_arr_set
    
    def finalize_prediction(self, patterns_arr_set):
        scores = []
        scores_each = []
        every_scores = []
        shape = np.array(self.dims, dtype=int)
        shape = shape[shape != 0]  
        layer = self.layers[-1]
        for patterns_arr_temp in tqdm(patterns_arr_set):
            learninggroup_scores = []
            learninggroup_scores_each = []
            learninggroup_every_scores = []
            for label in range(self.label_num):
                pattern_arr = patterns_arr_temp[label]
                score_temp, label_scores_each, every_scores_temp = self.process_ScoringLayer_finalize(label=label, shape=shape, patterns_arr=pattern_arr, layer=layer)
                learninggroup_scores.append(score_temp)
                learninggroup_scores_each.append(label_scores_each)
                learninggroup_every_scores.append(every_scores_temp)
            scores.append(learninggroup_scores)
            scores_each.append(learninggroup_scores_each)
            every_scores.append(learninggroup_every_scores)
        scores = np.array(scores)
        return scores, scores_each, every_scores
    
    def predict(self, x, save_options):
        print("\nModel inference started...")
        for label in range(self.label_num):
            scorectcam_shape = self.ctcams[label][-1].shape
            self.ctcams_sorted[label] = np.zeros(shape=scorectcam_shape, dtype=object)
            for index in np.ndindex(scorectcam_shape):
                self.ctcams_sorted[label][index] = self.dummy_ctcams[label].ctcam_sorting(self.ctcams[label][-1][index])
        scores = []
        scores_each = []
        every_scores = []
        for image_num in tqdm(range(len(x))):
            image_save = False
            image = np.array(x[image_num])
            if (image_num in save_options["save_image_indices"]):
                image_save = True
            if (image_save):
                image_name = "pred_"+str(image_num)+"image"
                save_image(images=image, save_dir=save_options["save_image_dir"], image_name=image_name, image_type=save_options["save_image_type"], one_file=save_options["save_image_one_file"], float_type=save_options["save_image_float"], use_hex=save_options["save_image_hex"])

            
            learninggroup_scores = []
            learninggroup_scores_each = []
            learninggroup_every_scores = []
            for label in range(self.label_num):
                image = np.array(x[image_num])
                score_temp = 0
                for layer_num, layer in enumerate(self.layers):
                    shape = np.array(self.dims[:layer_num], dtype=int)
                    shape = shape[shape != 0]  
                    

                    # JustConv2DLayer
                    if isinstance(layer, JustConv2DLayer):
                        image = self.process_JustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # MatchJustConv2DLayer
                    elif isinstance(layer, MatchJustConv2DLayer):
                        image = self.process_MatchJustConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Conv2DLayer
                    elif isinstance(layer, MatchConv2DLayer):
                        image = self.process_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Conv2DLayer_old
                    elif isinstance(layer, MatchConv2DLayer_old):
                        image = self.process_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # n_Conv2DLayer
                    elif isinstance(layer, n_MatchConv2DLayer):
                        image = self.process_n_MatchConv2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # n_Conv2DLayer_old
                    elif isinstance(layer, n_MatchConv2DLayer_old):
                        image = self.process_n_MatchConv2DLayer_old(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_JustConv2DLayer
                    elif isinstance(layer, Top_JustConv2DLayer):
                        image = self.process_Top_JustConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_MatchJustConv2DLayer
                    elif isinstance(layer, Top_MatchJustConv2DLayer):
                        image = self.process_Top_MatchJustConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_Conv2DLayer
                    elif isinstance(layer, Top_MatchConv2DLayer):
                        image = self.process_Top_MatchConv2DLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # Top_Conv2DLayer_old
                    elif isinstance(layer, Top_MatchConv2DLayer_old):
                        image = self.process_Top_MatchConv2DLayer_old_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # ZeroPaddingLayer
                    elif isinstance(layer, ZeroPaddingLayer):
                        image = self.process_ZeroPaddingLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # MaxPooling2DLayer
                    elif isinstance(layer, MaxPooling2DLayer):
                        image = self.process_MaxPooling2DLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # NormalizationLayer
                    elif isinstance(layer, NormalizationLayer):
                        image = self.process_NormalizationLayer(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    # ScoringLayer
                    elif isinstance(layer, ScoringLayer):
                        score_temp, label_scores_each, every_scores_temp = self.process_ScoringLayer_predict(label=label, shape=shape, image=image, layer_num=layer_num, layer=layer)
                    else:
                        pass
                    if (image_save):
                        image_name = "pred_"+str(label)+"LG_"+str(image_num)+"image_"+str(layer_num)+"layer"
                        save_image(images=image, save_dir=save_options["save_image_dir"], image_name=image_name, image_type=save_options["save_image_type"], one_file=save_options["save_image_one_file"], float_type=save_options["save_image_float"], use_hex=save_options["save_image_hex"])

                learninggroup_scores.append(score_temp)
                learninggroup_scores_each.append(label_scores_each)
                learninggroup_every_scores.append(every_scores_temp)
            scores.append(learninggroup_scores)
            scores_each.append(learninggroup_scores_each)
            every_scores.append(learninggroup_every_scores)
            if (image_num in save_options["save_score_indices"]):
                score_name = "score_"+str(image_num)+"image"
                save_score(scores=np.array(learninggroup_scores_each), save_dir=save_options["save_score_dir"], score_name=score_name, one_file=True)
        scores = np.array(scores)
        return scores, scores_each, every_scores
    
def save_top_filter_count(model: FLASCHE, save_dir, one_file):
    print("saving top filter ctcam counts...")
    for layer_num, layer in enumerate(model.layers):
        if (issubclass(type(layer), Top_ConvLayer)):
            if (one_file):
                file_path = save_dir+"/ctcam_layer{layernum}_top_filters_counts.txt".format(layernum=layer_num)
                with open(file_path,'w') as f:
                    f.truncate(0)
                    for i in range(model.label_num):
                        ctcams = model.ctcams[i][layer_num]
                        for index in np.ndindex(*ctcams.shape):
                            ctcam_info = "class{classnum}_layer{layernum}_ctcams{ind}\n".format(classnum=i, layernum=layer_num, ind=index)
                            f.write(ctcam_info)
                            f.write('rank   | pattern |  count  \n')
                            ctcam_temp = CTCAM(length=0, kernel_size=(0, 0)).ctcam_sorting(ctcam=ctcams[index])
                            #ctcam_temp = ctcams[index]
                            for rank, (pattern, count) in enumerate(ctcam_temp, start=1):
                                f.write(f'  {rank:<5}|   {pattern:<6}|  {count:<5}\n')
                            f.write('\n')
            else:
                for i in range(model.label_num):
                    ctcams = model.ctcams[i][layer_num]
                    for index in np.ndindex(*ctcams.shape):
                        file_path = save_dir+"/class{classnum}_layer{layernum}_ctcams{ind}.txt".format(classnum=i, layernum=layer_num, ind=index)
                        with open(file_path,'w') as f:
                            f.truncate(0)
                            f.write('rank   | pattern |  count  \n')
                            ctcam_temp = CTCAM(length=0, kernel_size=(0, 0)).ctcam_sorting(ctcam=ctcams[index])
                            #ctcam_temp = ctcams[index]
                            for rank, (pattern, count) in enumerate(ctcam_temp, start=1):
                                f.write(f'  {rank:<5}|   {pattern:<6}|  {count:<5}\n')

def save_trained_count(model: FLASCHE, save_dir, one_file):
    print("saving trained ctcam counts...")
    for layer_num, layer in enumerate(model.layers):
        if (isinstance(layer, ScoringLayer)):
            if (one_file):
                file_path = save_dir+"/ctcam_layer{layernum}_scoring_counts.txt".format(layernum=layer_num)
                with open(file_path,'w') as f:
                    f.truncate(0)
                    for i in range(model.label_num):
                        ctcams = model.ctcams[i][layer_num]
                        for index in np.ndindex(*ctcams.shape):
                            ctcam_info = save_dir+"/class{classnum}_layer{layernum}_ctcams{ind}\n".format(classnum=i, layernum=layer_num, ind=index)
                            f.write(ctcam_info)
                            f.write('rank   | pattern |  count  \n')
                            ctcam_temp = CTCAM(length=0, kernel_size=(0, 0)).ctcam_sorting(ctcam=ctcams[index])
                            #ctcam_temp = ctcams[index]
                            for rank, (pattern, count) in enumerate(ctcam_temp, start=1):
                                f.write(f'  {rank:<5}|   {pattern:<6}|  {count:<5}\n')
                            f.write('\n')
            else:
                for i in range(model.label_num):
                    ctcams = model.ctcams[i][layer_num]
                    for index in np.ndindex(*ctcams.shape):
                        file_path = save_dir+"/class_{classnum}_layer_{layernum}_ctcams_{ind}.txt".format(classnum=i, layernum=layer_num, ind=index)
                        with open(file_path,'w') as f:
                            f.truncate(0)
                            f.write('rank   | pattern |  count  \n')
                            ctcam_temp = CTCAM(length=0, kernel_size=(0, 0)).ctcam_sorting(ctcam=ctcams[index])
                            #ctcam_temp = ctcams[index]
                            for rank, (pattern, count) in enumerate(ctcam_temp, start=1):
                                f.write(f'  {rank:<5}|   {pattern:<6}|  {count:<5}\n')

def save_trained_count_sep(model: FLASCHE, save_dir, top):
    print("saving trained ctcam counts...")
    for j in range(top):
        file_path = save_dir+f"/top{j+1}.txt"
        with open(file_path,'w') as f:
            f.truncate(0)
            for label in range(model.label_num):
                ctcams = model.ctcams_sorted[label]
                for index in np.ndindex(*ctcams.shape):
                    f.write(f"{int(ctcams[index][j][0]):03X}\n")
                    #f.write(f'{ctcams[index][j][0]}\n')

# def save_trained_count_sep_72bits(model: FLASCHE, save_dir, top):
#     print("saving trained ctcam counts...")
#     for j in range(top):
#         file_path = save_dir+f"/top{j+1}.txt"
#         with open(file_path,'w') as f:
#             f.truncate(0)
#             for label in range(model.label_num):
#                 ctcams = model.ctcams_sorted[label]
#                 ctcams_per_image = ctcams.shape[-2] * ctcams.shape[-1]
#                 cnt = 0
#                 bin_str = ''
#                 for index in np.ndindex(*ctcams.shape):
#                     bin_str = format(ctcams[index][j][0], '09b') + bin_str
#                     cnt += 1
#                     if (cnt%8==0):
#                         val = int(bin_str, 2)          # 2진 문자열 → int
#                         hex_str = format(val, '018X')  # 72비트 → 18자리 HEX (대문자)
#                         f.write(f"{hex_str}\n")
#                         bin_str = ''
#                     elif (cnt==ctcams_per_image):
#                         val = int(bin_str, 2)          # 2진 문자열 → int
#                         hex_str = format(val, '018X')  # 72비트 → 18자리 HEX (대문자)
#                         f.write(f"{hex_str}\n")
#                         bin_str = ''
#                         cnt = 0