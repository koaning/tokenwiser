import h5py

def save_coefficients(classifier, filename):
    """Save the coefficients of a linear model into a .h5 file."""
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("coef",  data=classifier.coef_)
        hf.create_dataset("intercept",  data=classifier.intercept_)
        hf.create_dataset("classes", data=classifier.classes_)

def load_coefficients(classifier, filename):
    """Attach the saved coefficients to a linear model."""
    with h5py.File(filename, 'r') as hf:
        coef = hf['coef'][:]
        intercept = hf['intercept'][:]
        classes = hf['classes'][:]
    classifier.coef_ = coef
    classifier.intercept_ = intercept
    classifier.classes_ = classes
    
def flatten(nested):
    """Flatten a nested list."""
    return [item for li in nested for item in li]
