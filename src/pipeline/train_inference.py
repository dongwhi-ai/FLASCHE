from __future__ import annotations

import argparse

from src.cam.cam import CTCAM
from src.filter.flasche_filter import FLASCHEFilter
from src.layer.flasche_layer import *
from src.utils.io import *
from src.data.mnist import *
from src.data.digits import *
from src.model.flasche import FLASCHE, save_top_filter_count, save_trained_count
from src.utils.seed import seed_everything
from src.eval.eval import *
from src.utils.misc import *

def main():
    ap = argparse.ArgumentParser("train & inference")
    ap.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    random_seed = cfg.get("seed", 42)
    seed_everything(random_seed)  # 없는 경우엔 무시 가능
    
    data_path = cfg.get("path", {}).get('data')
    data_cfg = cfg.get("data", {})
    labeling = cfg.get("labeling", {})
    input_spec = cfg.get("input_spec", {})
    model_spec = cfg.get("model_spec", {}) 
    save_options = cfg.get("save_options", {})
    
    # Data
    data_use = data_cfg.get("use")
    if data_use=="MNIST":
        X_train, X_test, y_train, y_test = load_all_original_MNIST(data_dir=data_path)
    elif data_use=="MNIST_selection":
        X_train, X_test, y_train, y_test = data_selection_mnist(data_dir=data_path, train_size=data_cfg.get("train_cnt"), test_size=data_cfg.get("infer_cnt"), random_seed=random_seed)
    elif data_use=="digits":
        X_train, X_test, y_train, y_test = data_selection_digits(train_size=data_cfg.get("train_cnt"), test_size=data_cfg.get("infer_cnt"), random_seed=random_seed)

    # Model
    model = FLASCHE(label_num=labeling.get("label_num"), input_shape=(input_spec.get("height"), input_spec.get("width")))
    layers = model_spec.get("layers")
    layer_params = model_spec.get("layer_params")
    for i in range(len(layers)):
        layer_type = layers[i]
        if layer_type=="conv":
            model.add(JustConv2DLayer(**layer_params[i]))
        elif layer_type=="top_conv":
            model.add(Top_JustConv2DLayer(**layer_params[i]))
        elif layer_type=="zeropad":
            model.add(ZeroPaddingLayer(**layer_params[i]))
        elif layer_type=="maxpool":
            model.add(MaxPooling2DLayer(**layer_params[i]))
        elif layer_type=="norm":
            model.add(NormalizationLayer(**layer_params[i]))
        elif layer_type=="score":
            model.add(ScoringLayer(**layer_params[i]))
        else:
            raise

    # Train
    model.fit(x=X_train, y=y_train, save_options=save_options.get("train_save_options"))

    if (save_options.get("save_count")):
        save_top_filter_count(model=model, save_dir=save_options.get("save_count_dir"), one_file=True)
        save_trained_count(model=model, save_dir=save_options.get("save_count_dir"), one_file=True)
    else:
        pass

    print("\n--- Training done ---")

    # Inference
    scores, scores_each, every_scores = model.predict(x=X_test, save_options=save_options.get("test_save_options"))

    print("\n--- Infefence done ---")

    prediction = most_similar_group(scores=scores)
    probabilities = score_proba(scores=scores)
    # accuracy
    print("\n--------------- accuracy ---------------")
    accuracy = calc_accuracy(test_labels=y_test, most_similar_group=prediction)

    # precision, recall, f1 score, accuracy
    print("\n--------------- precision, recall, f1 score, accuracy ---------------")
    clf_report(y_test=y_test, prediction=prediction, class_mapping=labeling.get("class_mapping"))

    # auc
    print("\n--------------- AUC(Area Under the Curve) ---------------")
    calc_auc(y_test=y_test, probabilities=probabilities)

if __name__ == "__main__":
    main()