"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Upload, ImageIcon, Loader2 } from 'lucide-react';
import Image from "next/image";
import { useRouter } from "next/navigation";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getAvailableModels } from "@/lib/api-service";

export function UploadImagem() {
  const router = useRouter();
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | undefined>();
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  const [mostrarOpcoesAvancadas, setMostrarOpcoesAvancadas] = useState(false);
  const [usarDataset1, setUsarDataset1] = useState(false);
  const [usarDataset2, setUsarDataset2] = useState(false);
  const [selectedModelDataset1, setSelectedModelDataset1] = useState<string | undefined>();
  const [selectedModelDataset2, setSelectedModelDataset2] = useState<string | undefined>();
  const [availableModelsDataset1, setAvailableModelsDataset1] = useState<string[]>([]);
  const [availableModelsDataset2, setAvailableModelsDataset2] = useState<string[]>([]);

  useEffect(() => {
    async function loadModels() {
      if (mostrarOpcoesAvancadas) {
        setIsLoadingModels(true);
        try {
          const promises = [];

          if (usarDataset1) {
            promises.push(getAvailableModels(1));
          }

          if (usarDataset2) {
            promises.push(getAvailableModels(2));
          }

          if (promises.length > 0) {
            const results = await Promise.all(promises);

            if (usarDataset1) {
              setAvailableModelsDataset1(results[0]);
              if (!usarDataset2) setAvailableModelsDataset2([]);
            }

            if (usarDataset2) {
              const index = usarDataset1 ? 1 : 0;
              setAvailableModelsDataset2(results[index]);
              if (!usarDataset1) setAvailableModelsDataset1([]);
            }
          }
        } catch (error) {
          console.error("Erro ao carregar modelos:", error);
        } finally {
          setIsLoadingModels(false);
        }
      } else {
        setIsLoadingModels(true);
        try {
          const models = await getAvailableModels(3); // Assumindo que 3 é o ID para o endpoint default
          setAvailableModels(models);
        } catch (error) {
          console.error("Erro ao carregar modelos:", error);
        } finally {
          setIsLoadingModels(false);
        }
      }
    }

    loadModels();
  }, [mostrarOpcoesAvancadas, usarDataset1, usarDataset2]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setImageUrl(objectUrl);

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      setImageBase64(base64String);
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyzeImage = async () => {
    if (!imageBase64) return;

    if (mostrarOpcoesAvancadas) {
      if (usarDataset1 && !selectedModelDataset1) {
        alert("Selecione um modelo para o Dataset 1");
        return;
      }

      if (usarDataset2 && !selectedModelDataset2) {
        alert("Selecione um modelo para o Dataset 2");
        return;
      }

      if (!usarDataset1 && !usarDataset2) {
        alert("Selecione pelo menos um dataset");
        return;
      }
    } else if (!selectedModel) {
      alert("Selecione um modelo");
      return;
    }

    setIsAnalyzing(true);

    try {
      localStorage.setItem('analyzedImage', imageUrl || '');
      localStorage.setItem('imageBase64', imageBase64);
      localStorage.setItem('mostrarOpcoesAvancadas', mostrarOpcoesAvancadas.toString());

      if (mostrarOpcoesAvancadas) {
        localStorage.setItem('usarDataset1', usarDataset1.toString());
        localStorage.setItem('usarDataset2', usarDataset2.toString());

        if (usarDataset1) {
          localStorage.setItem('selectedModelDataset1', selectedModelDataset1 || '');
        }

        if (usarDataset2) {
          localStorage.setItem('selectedModelDataset2', selectedModelDataset2 || '');
        }
      } else {
        localStorage.setItem('selectedModel', selectedModel || '');
      }

      router.push('/ferramenta/analise');
    } catch (error) {
      console.error("Erro ao analisar imagem:", error);
      setIsAnalyzing(false);
    }
  };

  return (
      <div className="space-y-4">
        <div className="flex items-center justify-center">
          <label
              htmlFor="image-upload"
              className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 dark:hover:bg-gray-800 dark:bg-gray-900 border-gray-300 dark:border-gray-700 transition-all"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-10 h-10 mb-3 text-gray-400" />
              <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                <span className="font-semibold">Clique para fazer upload</span> ou
                arraste e solte
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Faça upload de uma imagem de situação de trânsito (PNG ou JPG)
              </p>
            </div>
            <input
                id="image-upload"
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleImageUpload}
            />
          </label>
        </div>

        {imageUrl && (
            <div className="flex flex-col items-center space-y-4">
              <div className="flex flex-wrap justify-center w-full gap-4">
                <Image
                    src={imageUrl || "/placeholder.svg"}
                    alt="Situação de trânsito"
                    width={350}
                    height={350}
                    className="object-contain rounded-lg shadow-md"
                />
              </div>

              <div className="flex flex-col gap-4 w-full">
                {!mostrarOpcoesAvancadas && (
                    <div className="flex items-center gap-2">
                      <Select
                          value={selectedModel}
                          onValueChange={setSelectedModel}
                          disabled={isLoadingModels || availableModels.length === 0}
                      >
                        <SelectTrigger className="w-[250px]">
                          <SelectValue placeholder={isLoadingModels ? "Carregando modelos..." : "Selecione o modelo"} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>Modelo</SelectLabel>
                            {availableModels.map(model => (
                                <SelectItem key={model} value={model}>{model}</SelectItem>
                            ))}
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </div>
                )}

                <div className="flex items-center gap-2">
                  <input
                      type="checkbox"
                      id="opcoes-avancadas"
                      checked={mostrarOpcoesAvancadas}
                      onChange={(e) => {
                        setMostrarOpcoesAvancadas(e.target.checked);
                        if (!e.target.checked) {
                          setUsarDataset1(false);
                          setUsarDataset2(false);
                          setSelectedModelDataset1(undefined);
                          setSelectedModelDataset2(undefined);
                        }
                      }}
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor="opcoes-avancadas" className="text-sm font-medium text-gray-700">
                    Mostrar opções avançadas
                  </label>
                </div>

                {mostrarOpcoesAvancadas && (
                    <div className="pl-6 space-y-4 border-l-2 border-gray-200">
                      <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="usar-dataset1"
                            checked={usarDataset1}
                            onChange={(e) => {
                              setUsarDataset1(e.target.checked);
                              if (!e.target.checked) {
                                setSelectedModelDataset1(undefined);
                              }
                            }}
                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="usar-dataset1" className="text-sm font-medium text-gray-700">
                          Usar Dataset 1
                        </label>

                        {usarDataset1 && (
                            <Select
                                value={selectedModelDataset1}
                                onValueChange={setSelectedModelDataset1}
                                disabled={isLoadingModels || availableModelsDataset1.length === 0}
                            >
                              <SelectTrigger className="w-[200px]">
                                <SelectValue placeholder={isLoadingModels ? "Carregando..." : "Selecione o modelo"} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectGroup>
                                  <SelectLabel>Modelo Dataset 1</SelectLabel>
                                  {availableModelsDataset1.map(model => (
                                      <SelectItem key={model} value={model}>{model}</SelectItem>
                                  ))}
                                </SelectGroup>
                              </SelectContent>
                            </Select>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="usar-dataset2"
                            checked={usarDataset2}
                            onChange={(e) => {
                              setUsarDataset2(e.target.checked);
                              if (!e.target.checked) {
                                setSelectedModelDataset2(undefined);
                              }
                            }}
                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="usar-dataset2" className="text-sm font-medium text-gray-700">
                          Usar Dataset 2
                        </label>

                        {usarDataset2 && (
                            <Select
                                value={selectedModelDataset2}
                                onValueChange={setSelectedModelDataset2}
                                disabled={isLoadingModels || availableModelsDataset2.length === 0}
                            >
                              <SelectTrigger className="w-[200px]">
                                <SelectValue placeholder={isLoadingModels ? "Carregando..." : "Selecione o modelo"} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectGroup>
                                  <SelectLabel>Modelo Dataset 2</SelectLabel>
                                  {availableModelsDataset2.map(model => (
                                      <SelectItem key={model} value={model}>{model}</SelectItem>
                                  ))}
                                </SelectGroup>
                              </SelectContent>
                            </Select>
                        )}
                      </div>
                    </div>
                )}

                <Button
                    onClick={handleAnalyzeImage}
                    disabled={
                        isAnalyzing ||
                        !imageUrl ||
                        (mostrarOpcoesAvancadas
                            ? ((!usarDataset1 && !usarDataset2) ||
                                (usarDataset1 && !selectedModelDataset1) ||
                                (usarDataset2 && !selectedModelDataset2))
                            : !selectedModel)
                    }
                    className="flex items-center space-x-2 bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white mt-4"
                >
                  {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Analisando...</span>
                      </>
                  ) : (
                      <>
                        <ImageIcon className="w-4 h-4" />
                        <span>Analisar Situação</span>
                      </>
                  )}
                </Button>
              </div>
            </div>
        )}
      </div>
  );
}