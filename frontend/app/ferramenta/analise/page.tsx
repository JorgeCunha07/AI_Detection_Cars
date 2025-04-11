"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { AnaliseResultado } from "@/components/analise-resultado";
import { ChatAssistente } from "@/components/chat-assistente";
import Tabs from "@/components/Tabs";
import Message from "@/app/types/Message";

export default function AnalisePage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("teste");
  const [isInitialized, setIsInitialized] = useState(false);

  const [mostrarOpcoesAvancadas, setMostrarOpcoesAvancadas] = useState(false);
  const [usarDataset1, setUsarDataset1] = useState(false);
  const [usarDataset2, setUsarDataset2] = useState(false);
  const [selectedModelDataset1, setSelectedModelDataset1] =
    useState<string>("");
  const [selectedModelDataset2, setSelectedModelDataset2] =
    useState<string>("");

  const tabs = [
    { id: "conversa", label: "Conversa" },
    { id: "quiz", label: "Quiz" },
    { id: "pesquisa", label: "Pesquisa" },
  ];

  const initialMessages: Message[] = [
    { id: "1", content: "Bem-vindo ao chat!", role: "assistant" },
    { id: "2", content: "Bem-vindo ao quiz!\nPara começar, escreva 'começar'", role: "assistant" },
    { id: "3", content: "Bem-vindo à pesquisa!", role: "assistant" },
  ];

  useEffect(() => {
    // Recuperar dados do localStorage
    const storedImage = localStorage.getItem("analyzedImage");
    const storedImageBase64 = localStorage.getItem("imageBase64");
    const storedMostrarOpcoesAvancadas =
      localStorage.getItem("mostrarOpcoesAvancadas") === "true";

    if (storedImage) setImageUrl(storedImage);
    if (storedImageBase64) setImageBase64(storedImageBase64);
    setMostrarOpcoesAvancadas(storedMostrarOpcoesAvancadas);

    if (storedMostrarOpcoesAvancadas) {
      const storedUsarDataset1 =
        localStorage.getItem("usarDataset1") === "true";
      const storedUsarDataset2 =
        localStorage.getItem("usarDataset2") === "true";

      setUsarDataset1(storedUsarDataset1);
      setUsarDataset2(storedUsarDataset2);

      if (storedUsarDataset1) {
        const storedModelDataset1 = localStorage.getItem(
          "selectedModelDataset1"
        );
        if (storedModelDataset1) setSelectedModelDataset1(storedModelDataset1);
      }

      if (storedUsarDataset2) {
        const storedModelDataset2 = localStorage.getItem(
          "selectedModelDataset2"
        );
        if (storedModelDataset2) setSelectedModelDataset2(storedModelDataset2);
      }
    } else {
      const storedModel = localStorage.getItem("selectedModel");
      if (true) setSelectedModel("teste");
    }

    setIsInitialized(true);
  }, []);

  // Se os dados ainda não foram carregados do localStorage
  if (!isInitialized) {
    return (
      <div className="container mx-auto py-8 px-4 flex justify-center items-center min-h-[50vh]">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  // Se não houver imagem, redirecionar para a página inicial da ferramenta
  if (!imageUrl && isInitialized) {
    return (
      <div className="container mx-auto py-8 px-4">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold text-gray-800">
            Nenhuma imagem analisada
          </h2>
          <p className="text-gray-600">
            Precisa de analisar uma imagem primeiro.
          </p>
          <Link href="/ferramenta">
            <Button className="bg-blue-600 hover:bg-blue-700">
              Voltar para upload de imagem
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6 flex justify-between items-center">
        <Link
          href="/ferramenta"
          className="flex items-center text-blue-600 hover:text-blue-800"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Voltar para upload
        </Link>
        <div className="text-sm text-gray-500">
          {mostrarOpcoesAvancadas ? (
            <span>
              Modo: <span className="font-medium">Avançado</span> |
              {usarDataset1 && (
                <span>
                  {" "}
                  Dataset 1:{" "}
                  <span className="font-medium">{selectedModelDataset1}</span>
                </span>
              )}
              {usarDataset1 && usarDataset2 && <span> | </span>}
              {usarDataset2 && (
                <span>
                  {" "}
                  Dataset 2:{" "}
                  <span className="font-medium">{selectedModelDataset2}</span>
                </span>
              )}
            </span>
          ) : (
            <span>
              Modelo: <span className="font-medium">{selectedModel}</span>
            </span>
          )}
        </div>
      </div>

      <Card className="w-full max-w-full mx-auto">
        <CardHeader>
          <CardTitle className="text-4xl text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 font-bold">
            Resultado da Análise
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Componente de Análise de Resultado */}
          <AnaliseResultado
            imageUrl={imageUrl}
            imageBase64={imageBase64}
            mostrarOpcoesAvancadas={mostrarOpcoesAvancadas}
            usarDataset1={usarDataset1}
            usarDataset2={usarDataset2}
            selectedModel="teste"
            selectedModelDataset1={selectedModelDataset1}
            selectedModelDataset2={selectedModelDataset2}
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
            <ChatAssistente
              initialMessages={[initialMessages[1]]}
              type="quiz"
            />
            <ChatAssistente
              initialMessages={[initialMessages[2]]}
              type="pesquisa"
            />
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}