"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import {
  analyzeImage,
  analyzeImageDefault,
  analyzeImageMultiDataset,
  type ImageAnalysisResult,
} from "@/lib/api-service";
import { DescriptionGenerator } from "./description-generator";
import { translateLabel } from "@/lib/utils";
import Tabs from "./Tabs";
import { ChatAssistente } from "./chat-assistente";
import Message from "@/app/types/Message";

interface AnaliseResultadoProps {
  imageUrl: string | null;
  imageBase64: string | null;
  mostrarOpcoesAvancadas: boolean;
  usarDataset1: boolean;
  usarDataset2: boolean;
  selectedModel?: string;
  selectedModelDataset1?: string;
  selectedModelDataset2?: string;
}

export function AnaliseResultado({
  imageUrl,
  imageBase64,
  mostrarOpcoesAvancadas,
  usarDataset1,
  usarDataset2,
  selectedModel,
  selectedModelDataset1,
  selectedModelDataset2,
}: AnaliseResultadoProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataset1Result, setDataset1Result] =
    useState<ImageAnalysisResult | null>(null);
  const [dataset2Result, setDataset2Result] =
    useState<ImageAnalysisResult | null>(null);
  const [defaultResult, setDefaultResult] =
    useState<ImageAnalysisResult | null>(null);

  const tabs = [
    { id: "conversa", label: "Conversa" },
    { id: "quiz", label: "Quiz" },
    { id: "pesquisa", label: "Pesquisa" },
  ];

  const initialMessages: Message[] = [
    { id: "1", content: "Bem-vindo ao chat!", role: "assistant" },
    {
      id: "2",
      content: "Bem-vindo ao quiz!\nPara começar, escreva 'começar'",
      role: "assistant",
    },
    { id: "3", content: "Bem-vindo à pesquisa!", role: "assistant" },
  ];

  useEffect(() => {
    async function fetchAnalysis() {
      if (!imageBase64) {
        setError("Imagem não disponível");
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        if (mostrarOpcoesAvancadas) {
          if (!usarDataset1 && !usarDataset2) {
            setError("Nenhum dataset selecionado");
            setIsLoading(false);
            return;
          }

          if (usarDataset1 && !selectedModelDataset1) {
            setError("Modelo para Dataset 1 não selecionado");
            setIsLoading(false);
            return;
          }

          if (usarDataset2 && !selectedModelDataset2) {
            setError("Modelo para Dataset 2 não selecionado");
            setIsLoading(false);
            return;
          }

          const results = await analyzeImageMultiDataset(imageBase64, {
            useDataset1: usarDataset1,
            useDataset2: usarDataset2,
            modelDataset1: selectedModelDataset1,
            modelDataset2: selectedModelDataset2,
          });

          if (results.dataset1) {
            setDataset1Result(results.dataset1);
          }

          if (results.dataset2) {
            setDataset2Result(results.dataset2);
          }
        } else {
          if (!selectedModel) {
            setError("Modelo não selecionado");
            setIsLoading(false);
            return;
          }

          const result = await analyzeImageDefault(imageBase64, selectedModel);
          setDefaultResult(result);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Erro ao analisar imagem"
        );
        console.error("Erro na análise:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchAnalysis();
  }, [
    imageBase64,
    mostrarOpcoesAvancadas,
    usarDataset1,
    usarDataset2,
    selectedModel,
    selectedModelDataset1,
    selectedModelDataset2,
  ]);

  function gerarDescricaoCombinada() {
    const allLabels = new Set<string>();
    let dataset1Count = 0;
    let dataset2Count = 0;

    if (dataset1Result) {
      dataset1Result.labels.forEach((label) => allLabels.add(label));
      dataset1Count = dataset1Result.labels.length;
    }

    if (dataset2Result) {
      dataset2Result.labels.forEach((label) => allLabels.add(label));
      dataset2Count = dataset2Result.labels.length;
    }

    const labelsArray = Array.from(allLabels);

    let description = `Esta imagem mostra uma situação de trânsito com ${labelsArray.join(
      ", "
    )}. `;

    if (dataset1Result && dataset2Result) {
      description += `O Dataset 1 identificou ${dataset1Count} elementos, enquanto o Dataset 2 
                     identificou ${dataset2Count} elementos. `;
    } else if (dataset1Result) {
      description += `O Dataset 1 identificou ${dataset1Count} elementos. `;
    } else if (dataset2Result) {
      description += `O Dataset 2 identificou ${dataset2Count} elementos. `;
    }

    description += `É importante estar atento a estes elementos para garantir uma condução segura.`;

    return description;
  }

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-8">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600 mb-4" />
        <p className="text-gray-600">Analisando imagem...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-600 rounded-lg">
        <p className="font-medium">Erro na análise:</p>
        <p>{error}</p>
      </div>
    );
  }

  if (
    mostrarOpcoesAvancadas &&
    ((usarDataset1 && dataset1Result) || (usarDataset2 && dataset2Result))
  ) {
    return (
      <div className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          {usarDataset1 && dataset1Result && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Dataset 1:</h3>
              <Image
                src={`data:image/jpeg;base64,${dataset1Result.image_base64}`}
                alt="Análise Dataset 1"
                width={500}
                height={350}
                className="object-contain rounded-lg shadow-md"
              />
              <div className="mt-4">
                <h4 className="font-semibold text-blue-600 mb-2">
                  Elementos detetados:
                </h4>
                <ul className="list-disc pl-5 space-y-1">
                  {dataset1Result.labels.map((label, index) => (
                    <li key={index}>
                      {translateLabel(label)}{" "}
                      {dataset1Result.counts && dataset1Result.counts[label] > 1
                        ? `(${dataset1Result.counts[label]})`
                        : ""}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {usarDataset2 && dataset2Result && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Dataset 2:</h3>
              <Image
                src={`data:image/jpeg;base64,${dataset2Result.image_base64}`}
                alt="Análise Dataset 2"
                width={500}
                height={350}
                className="object-contain rounded-lg shadow-md"
              />
              <div className="mt-4">
                <h4 className="font-semibold text-blue-600 mb-2">
                  Elementos detetados:
                </h4>
                <ul className="list-disc pl-5 space-y-1">
                  {dataset2Result.labels.map((label, index) => (
                    <li key={index}>
                      {translateLabel(label)}{" "}
                      {dataset2Result.counts && dataset2Result.counts[label] > 1
                        ? `(${dataset2Result.counts[label]})`
                        : ""}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
        <DescriptionGenerator
          results={
            dataset1Result && dataset2Result
              ? [dataset1Result, dataset2Result]
              : dataset1Result
              ? [dataset1Result]
              : dataset2Result
              ? [dataset2Result]
              : []
          }
        />
        {/* Componente de Chat */}
        <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 mb-4">
          Tire as suas dúvidas sobre esta situação:
        </h3>
        <Tabs tabs={tabs}>
          <ChatAssistente
            initialMessages={[initialMessages[0]]}
            type="conversa"
          />
          <ChatAssistente initialMessages={[initialMessages[1]]} type="quiz" />
          <ChatAssistente
            initialMessages={[initialMessages[2]]}
            type="pesquisa"
          />
        </Tabs>
      </div>
    );
  }

  if (!mostrarOpcoesAvancadas && defaultResult) {
    return (
      <div className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Imagem Original:</h3>
            {imageUrl && (
              <Image
                src={imageUrl || "/placeholder.svg"}
                alt="Situação de trânsito original"
                width={500}
                height={350}
                className="object-contain rounded-lg shadow-md"
              />
            )}
          </div>

          <div>
            <h3 className="text-xl font-semibold mb-4">Imagem Analisada:</h3>
            <Image
              src={`data:image/jpeg;base64,${defaultResult.image_base64}`}
              alt="Situação de trânsito analisada"
              width={500}
              height={350}
              className="object-contain rounded-lg shadow-md"
            />
          </div>
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-xl font-semibold mb-4 text-blue-700">
            Análise da Situação:
          </h3>
          <div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">
                Elementos detetados:
              </h4>
              <ul className="list-disc pl-5 space-y-1">
                {defaultResult.labels.map((label, index) => (
                  <li key={index}>
                    {translateLabel(label)}{" "}
                    {defaultResult.counts && defaultResult.counts[label] > 1
                      ? `(${defaultResult.counts[label]})`
                      : ""}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
        {defaultResult.labels && defaultResult.labels.length > 0 && (
          <DescriptionGenerator results={[defaultResult]} />
        )}

        {/* Componente de Chat */}
        <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 mb-4">
          Tire as suas dúvidas sobre esta situação:
        </h3>
        <Tabs tabs={tabs}>
          <ChatAssistente
            initialMessages={[initialMessages[0]]}
            type="conversa"
          />
          <ChatAssistente initialMessages={[initialMessages[1]]} type="quiz" />
          <ChatAssistente
            initialMessages={[initialMessages[2]]}
            type="pesquisa"
          />
        </Tabs>
      </div>
    );
  }

  return (
    <div className="p-4 bg-yellow-50 text-yellow-600 rounded-lg">
      <p className="font-medium">Nenhum resultado disponível</p>
      <p>Verifique as configurações e tente novamente.</p>
    </div>
  );
}
