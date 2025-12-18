from laplace import Laplace
from sklearn.metrics import classification_report
import medmnist
from medmnist import INFO


data_flag = "organmnist3d"
info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])


full_dataset = DataClass(split='train', download=True, size=64)
