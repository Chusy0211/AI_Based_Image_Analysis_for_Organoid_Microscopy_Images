import os
import model_pool

if __name__ == "__main__":
    test_folder = "../datasets/train"
    model_prefix = "checkpoints/2026-03-25-17-07-33"

    model_pool = model_pool.Fold5ModelPool(pool_size=1, model_prefix=model_prefix)

    normal = 0
    normal_err = 0
    abnormal = 0
    abnormal_err = 0
    for fname in os.listdir(test_folder):
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"{fname}")

        pred = model_pool.predict(f"{test_folder}/{fname}")

        label = 1 if "abnormal" in fname.lower() else 0
        if label == 1:
            abnormal += 1
            if label != pred:
                abnormal_err += 1
        else:
            normal += 1
            if label != pred:
                normal_err += 1

    print(f"normal:{normal} normal_err:{normal_err} abnormal:{abnormal} abnormal_err:{abnormal_err}")
