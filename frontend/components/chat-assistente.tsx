"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";
import Message from "@/app/types/Message";
import MessageContainer from "./message-container";
import {
  answerQuizQuestion,
  getNextQuizQuestion,
  sendChatMessage,
  sendSearchMessage,
} from "@/lib/api-service";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface ChatAssistenteProps {
  initialMessages?: Message[];
  type?: "conversa" | "quiz" | "pesquisa";
}

export function ChatAssistente({
  initialMessages = [],
  type = "conversa",
}: ChatAssistenteProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [quizTip, setQuizTip] = useState<Message | undefined>();
  const [quizStarted, setQuizStarted] = useState<boolean>(false);
  const [lastQuestion, setLastQuestion] = useState<string>("");
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedChatModel, setSelectedChatModel] = useState<
    string | undefined
  >();
  const [validationMethod, setValidationMethod] = useState<
    "sbert" | "transformer"
  >("sbert");

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleEnterSubmit = (e: any) => {
    if (e.key === "Enter" && !isLoading) {
      e.preventDefault(); // prevent newline
      e.currentTarget.form?.requestSubmit(); // trigger form submit
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    handleRequestByType(input);
  };

  const handleNextQuizQuestion = async () => {
    setIsLoading(true);

    const apiResponse = await getNextQuizQuestion().finally(() => {
      setIsLoading(false);
    });

    setMessages((prev) => [...prev, apiResponse[0], apiResponse[1]]);
    setQuizTip(apiResponse[2]);
    setLastQuestion(apiResponse[0].content);
  };

  const handleRequestByType = async (input: string) => {
    switch (type) {
      case "conversa":
        setIsLoading(true);
        const aiResponseConversa = await sendChatMessage(input).finally(() => {
          setIsLoading(false);
        });
        setMessages((prev) => [...prev, aiResponseConversa]);

        break;
      case "quiz":
        if (input.trim() === "dica" && quizTip !== undefined) {
          // pedir dica quiz
          setIsLoading(true);
          setMessages((prev) => [...prev, quizTip]);
          setIsLoading(false);
        } else if (!quizStarted && input.trim() === "começar") {
          // começar quiz
          setQuizStarted(true);
          handleNextQuizQuestion();
        } else if (quizStarted) {
          // responder quiz
          setIsLoading(true);
          const apiResponseToQuizQuestion = await answerQuizQuestion(
            lastQuestion,
            input,
            validationMethod // Adicione o método como parâmetro
          ).finally(() => {
            setIsLoading(false);
          });
          setMessages((prev) => [...prev, apiResponseToQuizQuestion]);

          handleNextQuizQuestion();
        }
        break;
      case "pesquisa":
        setIsLoading(true);
        const apiResponsePesquisa = await sendSearchMessage(input).finally(
          () => {
            setIsLoading(false);
          }
        );
        setMessages((prev) => [...prev, apiResponsePesquisa]);

        break;
      default:
        console.error("Invalid chat type");
    }
  };

  return (
    <div className="space-y-4">
      {type === "quiz" && (
        <div className="flex items-center space-x-2 mb-4">
          <Switch
            id="validation-method"
            checked={validationMethod === "transformer"}
            onCheckedChange={(checked) =>
              setValidationMethod(checked ? "transformer" : "sbert")
            }
          />
          <Label htmlFor="validation-method">
            Método: {validationMethod === "sbert" ? "BERT" : "Transformer"}
          </Label>
        </div>
      )}
      <MessageContainer messages={messages} isLoading={isLoading} />
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <Textarea
          onKeyDown={handleEnterSubmit}
          value={input}
          onChange={handleInputChange}
          placeholder="Faça uma pergunta sobre esta situação de trânsito..."
          className="flex-1 resize-none"
          rows={2}
        />
        <Button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="self-end bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white"
        >
          <Send className="w-4 h-4" />
        </Button>
      </form>
    </div>
  );
}
