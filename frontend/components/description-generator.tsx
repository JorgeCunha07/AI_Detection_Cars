import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { generateDescriptionSeq2seq } from "@/lib/api-service";
import GenerateDescriptionRequest from "@/app/types/GenerateDescriptionRequest";

export function DescriptionGenerator() {
  const [description, setDescription] = useState<String>();
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const defaultRequest: GenerateDescriptionRequest = {
    labels: ["sinal_de_stop", "céu_limpo", "dia", "zona_residencial"],
    mode: "topk",
    topk: 10,
    temperature: 1.5,
  };

  const generateDescription = async () => {
    setIsLoading(true);
    const apiResponse = await generateDescriptionSeq2seq(defaultRequest);
    setDescription(apiResponse);
    setIsLoading(false);
  };

  useEffect(() => {
    generateDescription();
  }, []);

  return (
    <div>
      <h4 className="font-semibold text-blue-600 mb-2">Descrição:</h4>
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
