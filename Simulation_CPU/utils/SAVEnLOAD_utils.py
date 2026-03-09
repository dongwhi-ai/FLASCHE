import numpy as np
from tqdm import tqdm
from PIL import Image
from CTCAM_utils import *
from LAYER_utils import *
from MODEL_utils import *

############################################################## Image save #####################################################
def save_image(images, save_dir, image_name, image_type='.txt', one_file=False, float_type=False, use_hex=False):
    print("saving images...")
    
    image_shape = images.shape
    image_lead_dim = image_shape[:-2]
    if (one_file):
        with open(save_dir+f"/{image_name}.txt", "w") as f:
            for index in tqdm(np.ndindex(*image_lead_dim)):
                f.write(f"# Image {index}\n") 
                image = images[index]
                for row in image:
                    if float_type:
                        line = ' '.join(f"{val:.4f}" for val in row)
                    elif use_hex:
                        line = ' '.join(f"{int(val):02X}" for val in row)  # 0x00 형식
                    else:
                        line = ' '.join(f"{int(val):3d}" for val in row)
                    f.write(line + "\n")
                f.write("\n")
    else:
        if (image_type=='.jpg'):
            for index in tqdm(np.ndindex(*image_lead_dim)):
                image = Image.fromarray(images[index])
                image.save(save_dir+f"/{image_name}_{index}.jpg")
                file_path = save_dir+f"/{image_name}_{index}.txt"
                with open(file_path, "w") as f:
                    for row in image:
                        if float_type:
                            line = ' '.join(f"{val:.4f}" for val in row)
                        elif use_hex:
                            line = ' '.join(f"{int(val):02X}" for val in row)
                        else:
                            line = ' '.join(f"{int(val):3d}" for val in row)
                        f.write(line + "\n")
        else:
            pass
    return None

def save_one_image(image, save_dir, image_name, one_file):
    if (one_file):
        with open(save_dir+f"/{image_name}.txt", "w") as f:
            np.savetxt(f, image, fmt="%2X", delimiter=" ")
    else:
        pass
    return None

############################################################## Score save #####################################################
def save_score(scores, save_dir, score_name, one_file):
    print("saving scores...")
    if (one_file):
        lg_num = scores.shape[0]
        with open(save_dir+f"/{score_name}.txt", "w") as f:
            for label in range(lg_num):
                f.write(f"# Label {label}\n") 
                for index in np.ndindex(scores.shape[1:]):
                    f.write(f"{scores[label][index]} ")
                f.write("\n") 
            f.write("\n") 
    else:
        pass

############################################################## Count save #####################################################
# Due to reference issues, count save functions are implemented in MODEL_utils.py


"""
############################################################## Count load #####################################################
def load_top8filter_count(label_num, load_dir):
    print("loading top8filter counts...")
    
    edge_detector = [MNISTEdgeDetector() for _ in range(label_num)]
    ctcam1 = [CTCAM() for _ in range(label_num)]
    four_ctcams = [ctcam1[i].make_curve_ctcam_array() for i in range(label_num)]

    # layer 1
    for label in tqdm(range(label_num)):
        for hvrl in range(len(four_ctcams[label])):
            if hvrl == 0:
                edge = 'h'
            elif hvrl == 1:
                edge = 'v'
            elif hvrl == 2:
                edge = 'r'
            else:
                edge = 'l'
            ctcam_num = []
            ctcam_cnt = []
            file_path = load_dir+"/four_ctcams{L}{E}.txt".format(L=label, E=edge)
    
            with open(file_path, 'r') as file:
                for _ in range(512):
                    line = file.readline().strip()
                    num = line.split()[0]       
                    cnt = line.split()[1]
                    ctcam_num.append(int(num))
                    ctcam_cnt.append(int(cnt))
    
            for rank, ctcam in enumerate(four_ctcams[label][hvrl], start=1):
                for i in range(len(ctcam_num)):
                    if ctcam.number == ctcam_num[i]:
                        ctcam.counter = ctcam_cnt[i]
    
    for label in range(label_num):
        ctcam1[label].filter_sorting(four_ctcams[label])
        ctcam1[label].top8_curve_filter(four_ctcams[label],edge_detector[label])
        
    return edge_detector, four_ctcams

def load_trained_CTCAM_count(label_num, load_dir):
    print("loading trained CTCAM counts...")
    
    ctcam2 = [CTCAM() for _ in range(label_num)]
    eight_ctcams = [ctcam2[i].make_ctcam_array8() for i in range(label_num)]
    
    # layer 2
    for label in tqdm(range(label_num)):
        for output in range(len(eight_ctcams[label])):
            for ctcams_low in range(len(eight_ctcams[label][output])):
                for ctcams_col in range(len(eight_ctcams[label][output][ctcams_low])):
                    ctcam_num = []
                    ctcam_cnt = []
                    file_path = load_dir+"/eight_ctcams{L},{O},{Cl},{Cc}.txt".format(L=label, O=output, Cl=ctcams_low, Cc=ctcams_col)
                    
                    with open(file_path,'r') as file:
                        for _ in range(512):
                            line = file.readline().strip()
                            num = line.split()[0]
                            cnt = line.split()[1]
                            ctcam_num.append(int(num))
                            ctcam_cnt.append(int(cnt))
    
                    for rank, ctcam in enumerate(eight_ctcams[label][output][ctcams_low][ctcams_col], start=1):
                        for i in range(len(ctcam_num)):
                            if ctcam.number == ctcam_num[i]:
                                ctcam.counter = ctcam_cnt[i]
    
    for label in range(label_num):
        ctcam2[label].sorting(eight_ctcams[label])
        
    return eight_ctcams

############################################################## Incremental learning ##############################################
def CTCAM_merge_top8filter(label_num, old_CTCAM, new_CTCAM, old_ratio, new_ratio):
    #print("merging CTCAM...")
    
    ratio_total = old_ratio+new_ratio
    old_ratio = old_ratio/ratio_total
    new_ratio = new_ratio/ratio_total
    print(f"merging CTCAM... (old:new = {old_ratio}:{new_ratio})")

    edge_detector = [MNISTEdgeDetector() for _ in range(label_num)]
    ctcam1 = [CTCAM() for _ in range(label_num)]
    merged_CTCAM = [ctcam1[i].make_curve_ctcam_array() for i in range(label_num)]
    
    for label in tqdm(range(label_num)): # L.G.
        for hvrl in range(len(new_CTCAM[label])): # 4
            for i in range(len(new_CTCAM[label][hvrl])): # 512
                number_temp = merged_CTCAM[label][hvrl][i].number
                for ctcam_new in new_CTCAM[label][hvrl]:
                    if (ctcam_new.number==number_temp):
                        count_new_temp = ctcam_new.counter
                        break
                for ctcam_old in old_CTCAM[label][hvrl]:
                    if (ctcam_old.number==number_temp):
                        count_old_temp = ctcam_old.counter
                        break
                merged_CTCAM[label][hvrl][i].counter = int(count_new_temp*new_ratio + count_old_temp*old_ratio)


        merged_CTCAM[label] = ctcam1[label].filter_sorting(four_ctcam=merged_CTCAM[label])
        merged_CTCAM[label] = ctcam1[label].top8_curve_filter(four_ctcam=merged_CTCAM[label],mnist_edge_detector=edge_detector[label])
    
    return edge_detector, merged_CTCAM


def CTCAM_merge_trained_CTCAM(label_num, old_CTCAM, new_CTCAM, old_ratio, new_ratio):
    #print("merging CTCAM...")
    
    ratio_total = old_ratio+new_ratio
    old_ratio = old_ratio/ratio_total
    new_ratio = new_ratio/ratio_total
    print(f"merging CTCAM... (old:new = {old_ratio}:{new_ratio})")

    ctcam2 = [CTCAM() for _ in range(label_num)]
    merged_CTCAM = [ctcam2[i].make_ctcam_array8() for i in range(label_num)]
    
    for label in tqdm(range(label_num)): # L.G.
        for output in range(len(new_CTCAM[label])): # 32
            for row in range(len(new_CTCAM[label][output])): # 4
                for col in range(len(new_CTCAM[label][output][row])): # 4
                    for i in range(len(new_CTCAM[label][output][row][col])): # 512
                        number_temp = merged_CTCAM[label][output][row][col][i].number
                        for ctcam_new in new_CTCAM[label][output][row][col]:
                            if (ctcam_new.number==number_temp):
                                count_new_temp = ctcam_new.counter
                                break
                        for ctcam_old in old_CTCAM[label][output][row][col]:
                            if (ctcam_old.number==number_temp):
                                count_old_temp = ctcam_old.counter
                                break
                        merged_CTCAM[label][output][row][col][i].counter = int(count_new_temp*new_ratio + count_old_temp*old_ratio)


        merged_CTCAM[label] = ctcam2[label].sorting(eight_ctcam=merged_CTCAM[label])
    return ctcam2, merged_CTCAM


"""