import { Loader2, RotateCw } from "lucide-react";
import { useEffect, useState } from "react";
import {
  generateDescriptionSeq2seq,
  ImageAnalysisResult,
} from "@/lib/api-service";
import GenerateDescriptionRequest from "@/app/types/GenerateDescriptionRequest";
import {
  getLabelPlural,
  isLabelValidForDescription,
  transformLabelsWithSpaces,
  translateLabel,
} from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { Button } from "./ui/button";

interface DescriptionGeneratorProps {
  results: ImageAnalysisResult[];
}

export function DescriptionGenerator({ results }: DescriptionGeneratorProps) {
  const [description, setDescription] = useState<String>();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [labels, setLabels] = useState<string[]>([]);
  const [mostrarOpcoesAvancadas, setMostrarOpcoesAvancadas] =
    useState<boolean>(false);

  const [selectedMode, setSelectedMode] = useState<string>("greedy");
  const [topkValue, setTopkValue] = useState<number>(3);
  const [topkTemperature, setTopkTemperature] = useState<number[]>([0.8]);
  const [beamWidth, setBeamWidth] = useState<number>(3);

  const availableModes = ["topk", "greedy", "beam"];

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
      mode: selectedMode,
      topk: topkValue,
      beam_width: beamWidth,
      temperature: topkTemperature[0],
    };

    if (!mostrarOpcoesAvancadas) request.mode = "greedy";

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
      <h3 className="text-xl font-semibold mb-4 text-blue-700">
        Descrição para [{labels.join(", ")}]:
      </h3>
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
          {mostrarOpcoesAvancadas && (
            <div className="mt-4 flex flex-col max-w-[200px]">
              <Select value={selectedMode} onValueChange={setSelectedMode}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder={"Selecione o método"} />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Modo</SelectLabel>
                    {availableModes.map((mode) => (
                      <SelectItem key={mode} value={mode}>
                        {mode}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
              {selectedMode === "topk" && (
                <div className="mt-4">
                  <p className="mt-2 text-sm">Topk number:</p>
                  <Input
                    value={topkValue}
                    onChange={(e) => setTopkValue(+e.target.value)}
                  />
                  <div className="flex flex-row items-center mt-4">
                    <div className="w-64">
                      <Slider
                        value={topkTemperature}
                        onValueChange={setTopkTemperature}
                        min={0.8}
                        max={2.0}
                        step={0.1}
                      />
                      <p className="mt-2 text-sm">
                        Temperature: {topkTemperature[0]}
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {selectedMode === "beam" && (
                <div className="mt-4">
                  <p className="mt-2 text-sm">Beam width:</p>
                  <Input
                    value={beamWidth}
                    onChange={(e) => setBeamWidth(+e.target.value)}
                  />
                </div>
              )}
              <Button onClick={generateDescription} className="mt-2">
                <RotateCw />
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
