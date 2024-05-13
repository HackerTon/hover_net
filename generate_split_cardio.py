import joblib
import random

if __name__ == "__main__":
    indices = [x for x in range(100)]
    random.shuffle(indices)
    train_indices = indices[:80]
    valid_indices = indices[80:]
    splits = [
        {
            "train": train_indices,
            "valid": valid_indices,
        }
    ]
    joblib.dump(splits, "splits.dat")