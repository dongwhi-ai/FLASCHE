import numpy as np
import itertools


####################################################### CTCAM class #########################################

class CTCAM():

  ################# CTCAM generation ####################
    def __init__(self, length, kernel_size):
        self.length = length
        self.kernel_size = kernel_size

    def make_ctcam(self):
        ctcam = np.zeros(shape=(self.length, 2), dtype=int)
        ctcam[:, 0] = np.arange(self.length)
        return ctcam
    
    def make_n_ctcam(self, n):
        ctcam = np.zeros(shape=(self.length, 2, n), dtype=int)
        ctcam[:, 0, :] = np.arange(self.length)
        return 

################ pattern detection ########################## 
    def pattern_detection_bi(self, quantized_image, pattern_size, strides, weights, output_size):
        patterns = np.zeros(shape=output_size, dtype=int)

        for i in range(output_size[0]):
            for j in range(output_size[0]):
                window = quantized_image[i*strides[0]:i*strides[0]+pattern_size[0], j*strides[1]:j*strides[1]+pattern_size[1]]
                pattern_temp = np.sum(window.flatten() * weights)
                patterns[i][j] = pattern_temp
        return patterns
    
    def pattern_detection_tri(self, quantized_image, pattern_size, strides, weights, output_size):
        patterns_arr = np.zeros(shape=output_size, dtype=object)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = quantized_image[i*strides[0]:i*strides[0]+pattern_size[0], j*strides[1]:j*strides[1]+pattern_size[1]]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j] = pattern_temp
        return patterns_arr
    
    def pattern_detection_nearn_bi(self, quantized_image, pattern_size, strides, weights, output_size):
        patterns_arr = np.zeros(shape=(*output_size, 5), dtype=int)
        padded_quantized_image = np.pad(array=quantized_image, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                window = quantized_image[i*strides[0]:i*strides[0]+pattern_size[0], j*strides[1]:j*strides[1]+pattern_size[1]]
    
                mid_y_start = i*strides[0]+1
                mid_y_end = i*strides[0]+pattern_size[0]+1
                mid_x_start = j*strides[1]+1
                mid_x_end = j*strides[1]+pattern_size[1]+1
                # mid
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
                patterns_arr[i][j][0] = np.sum(window.flatten() * weights)
                # top
                window = padded_quantized_image[mid_y_start-1:mid_y_end-1, mid_x_start:mid_x_end]
                patterns_arr[i][j][1] = np.sum(window.flatten() * weights)
                # bot
                window = padded_quantized_image[mid_y_start+1:mid_y_end+1, mid_x_start:mid_x_end]
                patterns_arr[i][j][2] = np.sum(window.flatten() * weights)
                # left
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start-1:mid_x_end-1]
                patterns_arr[i][j][3] = np.sum(window.flatten() * weights)
                # right
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start+1:mid_x_end+1]
                patterns_arr[i][j][4] = np.sum(window.flatten() * weights)

        return patterns_arr
    
    def pattern_detection_nearn_tri(self, quantized_image, pattern_size, strides, weights, output_size):
        patterns_arr = np.zeros(shape=(*output_size, 5), dtype=object)
        padded_quantized_image = np.pad(array=quantized_image, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        for i in range(output_size[0]):
            for j in range(output_size[1]):
                
                mid_y_start = i*strides[0]+1
                mid_y_end = i*strides[0]+pattern_size[0]+1
                mid_x_start = j*strides[1]+1
                mid_x_end = j*strides[1]+pattern_size[1]+1
                # mid
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j][0] = pattern_temp
                # top
                window = padded_quantized_image[mid_y_start-1:mid_y_end-1, mid_x_start:mid_x_end]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j][1] = pattern_temp
                # bot
                window = padded_quantized_image[mid_y_start+1:mid_y_end+1, mid_x_start:mid_x_end]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j][2] = pattern_temp
                # left
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start-1:mid_x_end-1]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j][3] = pattern_temp
                # right
                window = padded_quantized_image[mid_y_start:mid_y_end, mid_x_start+1:mid_x_end+1]
                two_positions = list(zip(*np.where(window==2)))
                num_twos = len(two_positions)
                all_combinations = itertools.product([0, 1], repeat=num_twos)
                pattern_temp = []
                for combination in all_combinations:
                    new_arr = window.copy()
                    for (pos, val) in zip(two_positions, combination):
                        new_arr[pos] = val
                    pattern_temp.append(np.sum(new_arr.flatten() * weights))
                patterns_arr[i][j][4] = pattern_temp

        return patterns_arr
    
    def pattern_counting(self, pattern, ctcam):
        ctcam[pattern, 1] += 1
        return None
    
    def patterns_counting(self, patterns, ctcam):
        for pattern in patterns.flat:
            ctcam[pattern, 1] += 1
        return None
    
    def patterns_counting_ctcam_creation(self, patterns, ctcam):
        for pattern in patterns.flat:
            if (np.any(ctcam[:, 0]==pattern)):
                pattern_ind = np.where(ctcam[:, 0]==pattern)
                ctcam[pattern_ind, 1] += 1
            else:
                ctcam = np.vstack((ctcam, np.array([[pattern, 1]])))
        ctcam = ctcam[np.argsort(ctcam[:, 0])]
        return ctcam

################### CTCAM sorting ##########################
    def ctcam_sorting(self, ctcam):
        ctcam_sorted = ctcam[np.lexsort((-ctcam[:, 0], -ctcam[:, 1]))]
            
        return ctcam_sorted



################### CTCAM Scoring #########################
    def calc_score(self, ctcams_sorted, patterns, exclude_zero, penalty_interval, thresholds, score_vals):
        ctcam_y, ctcam_x = ctcams_sorted.shape
        score = 0
        score_detailed = np.zeros(shape=ctcams_sorted.shape, dtype=int)
        penalty = 0
        for row in range(ctcam_y):
            for col in range(ctcam_x):
                pattern = patterns[row][col]
                ctcam_sorted = ctcams_sorted[row][col]
                if (exclude_zero):
                    if (ctcam_sorted[0][0]==0):
                        ctcam_sorted = np.vstack((ctcam_sorted[1:], ctcam_sorted[0]))
                else:
                    if not (pattern==0):
                        if (ctcam_sorted[0][0]==0):
                            penalty += penalty_interval
                    else:
                        pass
                pattern_idx = np.where(ctcam_sorted[:, 0] == pattern)[0]
                if (ctcam_sorted[pattern_idx, 1]==0): # count가 0이면 rank가 범위 안이어도 0점처리 되도록 함.
                    continue
                else:
                    rank = pattern_idx + 1
                    for (threshold, score_val) in zip(thresholds, score_vals):
                        if rank < threshold:
                            score += score_val
                            score_detailed[row, col] = score_val
                            break
        total_score = score - penalty
        return total_score, score_detailed
                        

####################### TOP n filter selection ##########################
    def pattern_to_filter(self, pattern, kernel_size, bits):
        num = bin(pattern)
        num = num[2:]
        num = num.zfill(bits)
        kernel = np.array(list(map(int, num))).reshape(kernel_size)

        return kernel
    
    def top_n_filters(self, ctcam, kernel_size, top_num, filter_weight_min):
        kernel_size_y, kernel_size_x = kernel_size
        bits = kernel_size_y * kernel_size_x
        top_filters = []
        for pattern, count in ctcam:
            kernel = self.pattern_to_filter(pattern=pattern, kernel_size=kernel_size, bits=bits)
            if (np.sum(kernel)<filter_weight_min):
                continue
            else:
                top_filters.append(kernel)
            if (len(top_filters)==top_num):
                break

        top_filters = np.array(top_filters)
        return top_filters

    def top_n_filters_ctcam_creation(self, ctcam, kernel_size, top_num, filter_weight_min):
        kernel_size_y, kernel_size_x = kernel_size
        bits = kernel_size_y * kernel_size_x
        top_filters =[]

        ctcam_index = 0
        pattern_temp = 0
        used_patterns = []
        while (len(top_filters)<top_num):
            if (ctcam_index < len(ctcam)):
                kernel = self.pattern_to_filter(pattern=ctcam[ctcam_index][0], kernel_size=kernel_size, bits=bits)
                if (np.sum(kernel)<filter_weight_min):
                    pass
                else:
                    top_filters.append(kernel)
                    used_patterns.append(ctcam[ctcam_index])
                ctcam_index += 1
            else:
                if (pattern_temp not in used_patterns):
                    kernel = self.pattern_to_filter(pattern=pattern_temp, kernel_size=kernel_size, bits=bits)
                    if (np.sum(kernel)<filter_weight_min):
                        pass
                    else:
                        top_filters.append(kernel)
                        used_patterns.append(pattern_temp)
                    pattern_temp += 1
                else:
                    pattern_temp += 1


        top_filters = np.array(top_filters)
        return top_filters