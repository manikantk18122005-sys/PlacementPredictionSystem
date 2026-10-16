import numpy as np
from sklearn.cluster import DBSCAN

def create_data():

    points = {
        'A' : (2,2),
        'B' : (2,3),
        'C' : (3,2),
        'D' : (3,3),
        'I': (4.4, 3),
         'E' : (8,8),
        'F' : (8,9),
        'G' : (9,8),
        'H' : (25,25)
    }

    names = list(points)

    X = np.array([points[name] for name in names])

    return names, X

def create_model(X, eps, min_samples):

    model = DBSCAN(eps=eps, min_samples=min_samples)

    model.fit(X)

    return model

def display_results(model, names):

    core_indices = set(model.core_sample_indices_)

    for i, (name, label) in enumerate(zip(names, model.labels_)):

     if i in core_indices:
         kind = "core"
     elif label == -1:
         kind = "noise"
     else:
         kind = "border"
     print(
         f"{name}:  "
         f"cluster={label:>2}  "
         f"-> {kind}"

     )

def main():
    names, X = create_data()

    model = create_model(X, eps=1.5, min_samples=3)

    display_results(model, names)

if __name__ == "__main__":
    main()

