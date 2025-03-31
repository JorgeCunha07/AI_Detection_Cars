import tensorflow as tf
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Definir antes de importar outras bibliotecas


# Teste com TensorFlow
print("TensorFlow - GPUs detectadas:")
print(tf.config.list_physical_devices('GPU'))

try:
    import torch
    # Teste com PyTorch
    print("\nPyTorch - GPU disponível:")
    print(torch.cuda.is_available())
except ImportError:
    print("\nPyTorch não está instalado.")

try:
    import GPUtil
    # Teste com GPUtil
    print("\nGPUtil - Utilização da GPU:")
    GPUtil.showUtilization()
except ImportError:
    print("\nGPUtil não está instalado.")
