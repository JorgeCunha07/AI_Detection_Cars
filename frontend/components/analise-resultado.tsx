"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2 } from 'lucide-react';
import { analyzeImage, analyzeImageDefault, analyzeImageMultiDataset, type ImageAnalysisResult } from "@/lib/api-service";

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
                                     selectedModelDataset2
                                 }: AnaliseResultadoProps) {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [dataset1Result, setDataset1Result] = useState<ImageAnalysisResult | null>(null);
    const [dataset2Result, setDataset2Result] = useState<ImageAnalysisResult | null>(null);
    const [defaultResult, setDefaultResult] = useState<ImageAnalysisResult | null>(null);

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
                        modelDataset2: selectedModelDataset2
                    });

                    if (results.dataset1) {
                        setDataset1Result(results.dataset1);
                    }

                    if (results.dataset2) {
                        setDataset2Result(results.dataset2);
                    }
                }
                else {
                    if (!selectedModel) {
                        setError("Modelo não selecionado");
                        setIsLoading(false);
                        return;
                    }

                    const result = await analyzeImageDefault(imageBase64, selectedModel);
                    setDefaultResult(result);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : "Erro ao analisar imagem");
                console.error("Erro na análise:", err);
            } finally {
                setIsLoading(false);
            }
        }

        fetchAnalysis();
    }, [imageBase64, mostrarOpcoesAvancadas, usarDataset1, usarDataset2, selectedModel, selectedModelDataset1, selectedModelDataset2]);

    function gerarDescricaoCombinada() {
        const allLabels = new Set<string>();
        let dataset1Count = 0;
        let dataset2Count = 0;

        if (dataset1Result) {
            dataset1Result.labels.forEach(label => allLabels.add(label));
            dataset1Count = dataset1Result.labels.length;
        }

        if (dataset2Result) {
            dataset2Result.labels.forEach(label => allLabels.add(label));
            dataset2Count = dataset2Result.labels.length;
        }

        const labelsArray = Array.from(allLabels);

        let description = `Esta imagem mostra uma situação de trânsito com ${labelsArray.join(', ')}. `;

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

    if (mostrarOpcoesAvancadas && ((usarDataset1 && dataset1Result) || (usarDataset2 && dataset2Result))) {
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
                                <h4 className="font-semibold text-blue-600 mb-2">Elementos detectados:</h4>
                                <ul className="list-disc pl-5 space-y-1">
                                    {dataset1Result.labels.map((label, index) => (
                                        <li key={index}>
                                            {label} {dataset1Result.counts && dataset1Result.counts[label] > 1 ? `(${dataset1Result.counts[label]})` : ''}
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
                                <h4 className="font-semibold text-blue-600 mb-2">Elementos detectados:</h4>
                                <ul className="list-disc pl-5 space-y-1">
                                    {dataset2Result.labels.map((label, index) => (
                                        <li key={index}>
                                            {label} {dataset2Result.counts && dataset2Result.counts[label] > 1 ? `(${dataset2Result.counts[label]})` : ''}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    )}
                </div>

                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <h3 className="text-xl font-semibold mb-4 text-blue-700">Análise Combinada:</h3>
                    <p>{gerarDescricaoCombinada()}</p>
                </div>
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
                    <h3 className="text-xl font-semibold mb-4 text-blue-700">Análise da Situação:</h3>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div>
                            <h4 className="font-semibold text-blue-600 mb-2">Elementos detectados:</h4>
                            <ul className="list-disc pl-5 space-y-1">
                                {defaultResult.labels.map((label, index) => (
                                    <li key={index}>
                                        {label} {defaultResult.counts && defaultResult.counts[label] > 1 ? `(${defaultResult.counts[label]})` : ''}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div>
                            <h4 className="font-semibold text-blue-600 mb-2">Descrição:</h4>
                            <p className="text-gray-700">
                                {defaultResult.labels && defaultResult.labels.length > 0
                                    ? `Esta imagem mostra uma situação de trânsito com ${defaultResult.labels.join(', ')}. 
                     É importante estar atento a estes elementos para garantir uma condução segura.`
                                    : 'Não foi possível gerar uma descrição para esta imagem.'}
                            </p>
                        </div>
                    </div>
                </div>
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