import GenerateDescriptionRequest from "@/app/types/GenerateDescriptionRequest";
import Message from "@/app/types/Message";

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
    const base64Data = imageBase64.includes("base64,")
      ? imageBase64.split("base64,")[1]
      : imageBase64;

    const response = await fetch(`/api/proxy/image/findLabels/${datasetId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelName,
        image_base64: base64Data,
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
    const base64Data = imageBase64.includes("base64,")
      ? imageBase64.split("base64,")[1]
      : imageBase64;

    const response = await fetch(`/api/proxy/image/findLabels/3`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelName,
        image_base64: base64Data,
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
    const base64Data = imageBase64.includes("base64,")
      ? imageBase64.split("base64,")[1]
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

export async function sendChatMessage(message: string): Promise<Message> {
  try {
    const response = await fetch(`/api/proxy/chatbot/conversa`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    return {
      id: (Date.now() + 1).toString(),
      content: data.response,
      role: "assistant",
    } as Message;
  } catch (error) {
    console.error("Erro:", error);

    return {
      id: (Date.now() + 1).toString(),
      content: "Desculpe, ocorreu um erro ao processar sua mensagem.",
      role: "assistant",
    } as Message;
  }
}

export async function sendSearchMessage(message: string): Promise<Message> {
  try {
    const response = await fetch(`/api/proxy/chatbot/pesquisa`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        pergunta: message,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    return {
      id: (Date.now() + 1).toString(),
      content: `${data.artigo_encontrado}\n\n${data.conteudo}`,
      role: "assistant",
    } as Message;
  } catch (error) {
    console.error("Erro:", error);

    return {
      id: (Date.now() + 1).toString(),
      content: "Desculpe, ocorreu um erro ao processar sua mensagem.",
      role: "assistant",
    } as Message;
  }
}

export async function getNextQuizQuestion(): Promise<Message[]> {
  try {
    const response = await fetch(`/api/proxy/chatbot/quiz/pergunta`);

    if (!response.ok) {
      throw new Error(`Erro ao obter pergunta: ${response.status}`);
    }

    const data = await response.json();

    return [
      {
        id: (Date.now() + 1).toString(),
        content: data.pergunta,
        role: "assistant",
      },
      {
        id: (Date.now() + 2).toString(),
        content: "Escreva 'dica' para ver as respostas corretas.",
        role: "assistant",
      },
      {
        id: (Date.now() + 3).toString(),
        content: `-${data.respostas_corretas.join("\n-")}`,
        role: "assistant",
      },
    ];
  } catch (error) {
    console.error("Erro ao obter a próxima questão:", error);
    throw error;
  }
}

export async function answerQuizQuestion(
  question: string,
  answer: string,
  metodo: string
): Promise<Message> {
  try {
    const response = await fetch(`/api/proxy/chatbot/quiz/responder`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        pergunta: question,
        resposta: answer,
        metodo: metodo
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    return {
      id: (Date.now() + 1).toString(),
      content: `${data.correto === true ? "✅ Correto\n" : "❌ Incorreto\n"}${
        data.metodo ? "\nmétodo: " + data.metodo : ""
      }${
        data.respostas_aceites
          ? "\nRespostas corretas:\n-" + data.respostas_aceites.join("\n-")
          : ""
      }`,
      role: "assistant",
    } as Message;
  } catch (error) {
    console.error("Erro:", error);

    return {
      id: (Date.now() + 1).toString(),
      content: "Desculpe, ocorreu um erro ao processar a sua mensagem.",
      role: "assistant",
    } as Message;
  }
}

export async function generateDescriptionSeq2seq(
  request: GenerateDescriptionRequest
) {
  try {
    const response = await fetch(`/api/proxy/description/generate/seq2seq`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    return data.sentence;
  } catch (error) {
    console.error("Erro:", error);

    return "Erro ao gerar frase";
  }
}
