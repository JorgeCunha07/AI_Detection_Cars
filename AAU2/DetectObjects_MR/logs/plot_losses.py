import json
import matplotlib.pyplot as plt
import os

# Número de batches por época
NUM_BATCHES = 438

def process_log_file(filepath, num_batches=NUM_BATCHES):
    """
    Lê o arquivo JSON de log e retorna duas listas:
      - epochs: números das épocas.
      - total_losses: valor da loss total (loss * num_batches) para cada época.
    O arquivo pode conter a chave "train_loss" ou "avg_loss".
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    epochs = []
    total_losses = []

    # Verifica qual chave usar
    loss_key = None
    if len(data) > 0:
        if "train_loss" in data[0]:
            loss_key = "train_loss"
        elif "avg_loss" in data[0]:
            loss_key = "avg_loss"
        else:
            raise ValueError(f"Arquivo {filepath} não possui nem 'train_loss' nem 'avg_loss'.")

    for entry in data:
        epoch = entry.get("epoch")
        loss = entry.get(loss_key)
        if epoch is not None and loss is not None:
            epochs.append(epoch)
            total_losses.append(loss * num_batches)

    return epochs, total_losses

def plot_losses(filepaths, model_names, num_batches=NUM_BATCHES):
    """
    Para cada arquivo em filepaths, processa o log e gera um gráfico em preto e branco,
    usando diferentes estilos de linha e diferentes marcadores para cada modelo.
    :param filepaths: lista de caminhos para os arquivos JSON de log.
    :param model_names: lista com os nomes dos modelos (para a legenda) correspondentes aos arquivos.
    :param num_batches: número de batches (438) para multiplicar o loss.
    """
    plt.figure(figsize=(12, 8))

    # Estilos de linha e marcadores para diferenciar os modelos
    line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    marker_styles = ['o', 's', '^', 'D', 'v', '*']

    for fp, model, ls, marker in zip(filepaths, model_names, line_styles, marker_styles):
        try:
            epochs, total_losses = process_log_file(fp, num_batches)
            # Usa cor 'k' (preto) para a linha, mas marca cada ponto com um marcador distinto
            plt.plot(epochs, total_losses, marker=marker, linestyle=ls, color='k', label=model)
        except Exception as e:
            print(f"Erro ao processar {fp}: {e}")

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss por Epoch")
    plt.title("Loss Evolution for different models per epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Se preferir salvar o gráfico, descomente a linha abaixo:
    # plt.savefig("loss_comparison.png", dpi=300)

if __name__ == '__main__':
    # Lista de arquivos JSON de log – ajuste conforme necessário.
    filepaths = [
        "training_log_mobilenet.json",
        "training_log_vgg.json",
        "training_log_resnet.json",
        "training_log_simplecnn.json",
        "training_log_midcnn.json",
        "training_log_deepcnn.json"
    ]
    model_names = [
        "MobileNet",
        "VGG",
        "ResNet",
        "SimpleCNN",
        "MidCNN",
        "DeepCNN"
    ]

    # Verifica se os arquivos existem
    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"Aviso: Arquivo {fp} não foi encontrado.")

    plot_losses(filepaths, model_names)
