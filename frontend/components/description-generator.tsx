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

interface DescriptionGeneratorProps {
  results: ImageAnalysisResult[];
}

export function DescriptionGenerator({ results }: DescriptionGeneratorProps) {
  const [description, setDescription] = useState<String>();
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [labels, setLabels] = useState<string[]>([]);

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
    setIsLoading(true);

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
        <p className="text-gray-700">{description}</p>
      )}
    </div>
  );
}
