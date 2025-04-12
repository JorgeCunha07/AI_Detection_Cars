import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
  generateDescriptionSeq2seq,
  ImageAnalysisResult,
} from "@/lib/api-service";
import GenerateDescriptionRequest from "@/app/types/GenerateDescriptionRequest";
import {
  filterLabelsForDescription,
  getLabelPlural,
  isLabelValidForDescription,
  transformLabelsWithSpaces,
  translateLabel,
} from "@/lib/utils";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "./ui/select";

interface DescriptionGeneratorProps {
  results: ImageAnalysisResult[];
}

export function DescriptionGenerator({ results }: DescriptionGeneratorProps) {
  const [description, setDescription] = useState<String>();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [labels, setLabels] = useState<string[]>([]);
  const [mostrarOpcoesAvancadas, setMostrarOpcoesAvancadas] =
    useState<boolean>(false);
const [selectedMode, setSelectedMode] = useState<string>("topk");

  const mergeLabelsAnalyzed = () => {
    const labels: string[] = [];
    const translatedLabels = new Map<string, number>();

    if (results != null) {
      results.forEach((result) => {
        result.labels.forEach((label) => {
          const translatedLabel = translateLabel(label);
          if (isLabelValidForDescription(translatedLabel)) {
            translatedLabels.set(
              translateLabel(translatedLabel),
              result.counts[label]
            );
          }
        });
      });

      translatedLabels.keys().forEach((key) => {
        const counts = translatedLabels.get(key);

        if (counts !== undefined && counts > 1) {
          labels.push(getLabelPlural(key));
        } else {
          labels.push(key);
        }
      });
    }

    const transformedLabels = transformLabelsWithSpaces(labels);
    return transformedLabels;
  };

  const generateDescription = async () => {
    const request: GenerateDescriptionRequest = {
      labels: labels,
      mode: "topk",
      temperature: 1.5,
    };

    const apiResponse = await generateDescriptionSeq2seq(request);
    setDescription(apiResponse);
    setIsLoading(false);
  };

  useEffect(() => {
    setIsLoading(true);

    setLabels(mergeLabelsAnalyzed());

    if (labels.length > 0) {
      generateDescription();
    }
  }, [results]);

  return (
    <div className="mt-6 p-4 bg-blue-50 rounded-lg">
      <h3 className="text-xl font-semibold mb-4 text-blue-700">Descrição:</h3>
      {isLoading && (
        <div className="flex justify-start">
          <div className="max-w-[80%] p-3 rounded-lg bg-secondary text-secondary-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
          </div>
        </div>
      )}
      {description && !isLoading && (
        <>
          <p className="text-gray-700">{description}</p>
          <div className="flex items-center gap-2 mt-4">
            <input
              type="checkbox"
              id="opcoes-avancadas"
              checked={mostrarOpcoesAvancadas}
              onChange={(e) => {
                setMostrarOpcoesAvancadas(e.target.checked);
              }}
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <label
              htmlFor="opcoes-avancadas"
              className="text-sm font-medium text-gray-700"
            >
              Mostrar opções avançadas
            </label>
          </div>
          {/* {mostrarOpcoesAvancadas && (
            <Select
              value={selectedModelDataset1}
              onValueChange={setSelectedModelDataset1}
              disabled={isLoadingModels || availableModelsDataset1.length === 0}
            >
              <SelectTrigger className="w-[200px]">
                <SelectValue
                  placeholder={
                    isLoadingModels ? "Carregando..." : "Selecione o modelo"
                  }
                />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Modelo Dataset 1</SelectLabel>
                  {availableModelsDataset1.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          )} */}
        </>
      )}
    </div>
  );
}
