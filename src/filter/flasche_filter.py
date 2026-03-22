import numpy as np

class FLASCHEFilter():
    def __init__(self):
        pass

    def apply_function_and_stack(func, *args):
        return np.stack(list(map(func, *args)))
    
    def data_size(input_size, kernel_size, strides):
        height, width = input_size
        filter_size_y, filter_size_x = kernel_size
        strides[0], strides[1] = strides
        y_quotient, y_remainder = divmod((height-filter_size_y), strides[0])
        output_size_y = y_quotient + 1
        x_quotient, x_remainder = divmod((width-filter_size_x), strides[1])
        output_size_x = x_quotient + 1
        return (output_size_y, output_size_x)


    def normalize(self, image, stretching, new_max, old_min, old_max):
        if (old_max == 0):
            return image
        else:
            if (stretching):
                interval_old = old_max - old_min
                zeros = (image==0)
                offset_adjusted_image = image - old_min
                offset_adjusted_image[zeros] = 0
                norm_image = ((offset_adjusted_image / interval_old) * new_max).astype(int)
            else:
                norm_image = ((image * new_max) / old_max).astype(int)
        return norm_image

    def quantization_bi(self, input_image, threshold):
        quantized_image = np.zeros_like(input_image, dtype=int)
        flag = (input_image >= threshold)
        quantized_image[flag] = 1
        
        return quantized_image

    def quantization_tri(self, input_image, threshold1, threshold2):
        quantized_image = np.zeros_like(input_image, dtype=int)
        flag = (input_image >= threshold1)
        quantized_image[flag] = 1
        flag = (input_image >= threshold2) & (input_image < threshold1)
        quantized_image[flag] = 2
        
        return quantized_image

    def patch_matcher_bi(self, input_patch, kernel):
        if (input_patch==kernel):
            return 1
        else:
            return 0
                
    
    def patch_matcher_tri(self, input_patch, kernel):
        height, width = input_patch.shape
        match_flag = 1
        for i in range(height):
            for j in range(width):
                if not (input_patch[i][j]==kernel[i][j] or input_patch[i][j]==2):
                    match_flag = 0
                    break
            if not (match_flag):
                break
                
        return match_flag
    
    def patch_matcher_tri_together(self, input_patch, kernel):
        patch_0 = input_patch.copy()
        patch_0[patch_0==2] = 0
        patch_1 = input_patch.copy()
        patch_1[patch_1==2] = 1
        if (np.array_equal(patch_0, kernel) or np.array_equal(patch_1, kernel)):
            return 1
        else:
            return 0

    def filter_matcher_bi(self, quantized_image, kernel, strides, kernel_size, output_size):
        matched_image = np.zeros(shape=output_size, dtype=int)
        
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = quantized_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                matched_image[i][j] = self.patch_matcher_bi(input_patch=window, kernel=kernel)

        return matched_image
    
    def filter_matcher_tri(self, quantized_image, kernel, strides, kernel_size, output_size):
        matched_image = np.zeros(shape=output_size, dtype=int)
        
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = quantized_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                matched_image[i][j] = self.patch_matcher_tri(input_patch=window, kernel=kernel)

        return matched_image
    
    def filter_matcher_tri_together(self, quantized_image, kernel, strides, kernel_size, output_size):
        matched_image = np.zeros(shape=output_size, dtype=int)
        
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = quantized_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                matched_image[i][j] = self.patch_matcher_tri_together(input_patch=window, kernel=kernel)

        return matched_image

    def n_filter_matcher_bi(self, quantized_image, kernels, strides, kernel_size, output_size):
        n_matched_image = []
        for i in range(len(kernels)):
            matched_image = self.filter_matcher_bi(quantized_image=quantized_image, kernel=kernels[i], strides=strides, kernel_size=kernel_size, output_size=output_size)
            n_matched_image.append(matched_image)
        n_matched_image = np.stack(n_matched_image)
        return n_matched_image
    
    def n_filter_matcher_tri(self, quantized_image, kernels, strides, kernel_size, output_size):
        n_matched_image = []
        for i in range(len(kernels)):
            matched_image = self.filter_matcher_tri(quantized_image=quantized_image, kernel=kernels[i], strides=strides, kernel_size=kernel_size, output_size=output_size)
            n_matched_image.append(matched_image)
        n_matched_image = np.stack(n_matched_image)
        return n_matched_image
    
    def n_filter_matcher_tri_together(self, quantized_image, kernels, strides, kernel_size, output_size):
        n_matched_image = []
        for i in range(len(kernels)):
            matched_image = self.filter_matcher_tri_together(quantized_image=quantized_image, kernel=kernels[i], strides=strides, kernel_size=kernel_size, output_size=output_size)
            n_matched_image.append(matched_image)
        n_matched_image = np.stack(n_matched_image)
        return n_matched_image

    def just_convolution(self, input_image, kernel, strides, kernel_size, output_size):
        filtered_image = np.zeros(shape=output_size, dtype=int)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = input_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                convolution_result = np.sum(window * kernel)
                filtered_image[i][j] = convolution_result
                
        return filtered_image
    
    def match_just_convolution(self, input_image, matched_image, kernel, strides, scale, kernel_size, output_size):
        filtered_image = np.zeros(shape=output_size, dtype=int)
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = input_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                convolution_result = np.sum(window * kernel)
                if (matched_image[i][j]):
                    filtered_image[i][j] = convolution_result * scale
                else:
                    filtered_image[i][j] = convolution_result

        return filtered_image
    
    def match_convolution(self, input_image, matched_image, kernel, strides, kernel_size, output_size):
        filtered_image = np.zeros(shape=output_size, dtype=int)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                convolution_result = 0
                if (matched_image[i][j]):
                    window = input_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                    convolution_result = np.sum(window * kernel)
                filtered_image[i][j] = convolution_result

        return filtered_image

    def n_match_convolution(self, input_image, matched_images, kernels, strides, kernel_size, output_size):
        conv_num = len(kernels)
        filtered_image = np.zeros(shape=output_size, dtype=int)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                convolution_result = 0
                for k in range(conv_num):
                    if (matched_images[k][i][j]):
                        window = input_image[i*strides[0]:i*strides[0]+kernel_size[0], j*strides[1]:j*strides[1]+kernel_size[1]]
                        convolution_result = np.sum(window * kernels[k])
                        break
                filtered_image[i][j] = convolution_result

        return filtered_image


    ##################################################### zero padding ##################################################

    def zero_padding(self, input_image, pad_width):
        pad_width = tuple(tuple(x) for x in pad_width)
        pad_width = ((0, 0),) * (input_image.ndim-2) + pad_width
        padded_image = np.pad(array=input_image, pad_width=pad_width, mode='constant', constant_values=0)
        return padded_image
    
    ##################################################### max pooling ##################################################

    def max_pooling(self, input_image, strides, pool_size, output_size):
        pooled_image = np.zeros(shape=output_size, dtype=int)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = input_image[i*strides[0]:i*strides[0]+pool_size[0], j*strides[1]:j*strides[1]+pool_size[1]]
                pooled_image[i][j] = np.max(window)
        return pooled_image
