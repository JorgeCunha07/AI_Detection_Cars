// Tipos para as mensagens
export type MessageRole = "user" | "assistant";

export type Message = {
  role: MessageRole;
  content: string;
};

export type ImageAnalysisResult = {
  labels: string[];
  counts: Record<string, number>;
  image_base64: string;
};

export async function analyzeImage(
    imageBase64: string,
    modelName: string,
    datasetId: number | null = 1
): Promise<ImageAnalysisResult> {
  try {
    // Remove o prefixo "data:image/jpeg;base64," se existir
    const base64Data = imageBase64.includes('base64,')
        ? imageBase64.split('base64,')[1]
        : imageBase64;

    const response = await fetch(`/api/proxy/image/findLabels/${datasetId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelName,
        image_base64: base64Data
      }),
    });

    if (!response.ok) {
      throw new Error(`Erro na análise: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Erro ao analisar imagem:", error);
    throw error;
  }
}

export async function analyzeImageDefault(
    imageBase64: string,
    modelName: string
): Promise<ImageAnalysisResult> {
  try {
    const base64Data = imageBase64.includes('base64,')
        ? imageBase64.split('base64,')[1]
        : imageBase64;

    const response = await fetch(`/api/proxy/image/findLabels/3`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelName,
        image_base64: base64Data
      }),
    });

    if (!response.ok) {
      throw new Error(`Erro na análise: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Erro ao analisar imagem:", error);
    throw error;
  }
}

export async function analyzeImageMultiDataset(
    imageBase64: string,
    options: {
      useDataset1: boolean;
      useDataset2: boolean;
      modelDataset1?: string;
      modelDataset2?: string;
    }
): Promise<{
  dataset1?: ImageAnalysisResult;
  dataset2?: ImageAnalysisResult;
}> {
  try {
    const base64Data = imageBase64.includes('base64,')
        ? imageBase64.split('base64,')[1]
        : imageBase64;

    const promises = [];

    if (options.useDataset1 && options.modelDataset1) {
      promises.push(analyzeImage(imageBase64, options.modelDataset1, 1));
    }

    if (options.useDataset2 && options.modelDataset2) {
      promises.push(analyzeImage(imageBase64, options.modelDataset2, 2));
    }

    const results = await Promise.all(promises);

    const response: {
      dataset1?: ImageAnalysisResult;
      dataset2?: ImageAnalysisResult;
    } = {};

    let index = 0;

    if (options.useDataset1 && options.modelDataset1) {
      response.dataset1 = results[index++];
    }

    if (options.useDataset2 && options.modelDataset2) {
      response.dataset2 = results[index];
    }

    return response;
  } catch (error) {
    console.error("Erro ao analisar imagem:", error);
    throw error;
  }
}

export async function getAvailableModels(datasetId: number): Promise<string[]> {
  try {
    const response = await fetch(`/api/proxy/image/models/${datasetId}`);

    if (!response.ok) {
      throw new Error(`Erro ao buscar modelos: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Erro ao buscar modelos disponíveis:", error);
    throw error;
  }
}

export async function sendChatMessage(message: string, history: Message[]): Promise<string> {
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) {
      throw new Error("Failed to send message");
    }

    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error("Error sending message:", error);
    throw error;
  }
}