"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";
import Message from "@/app/types/Message";
import MessageContainer from "./message-container";

interface ChatAssistenteProps {
    initialMessages?: Message[];
    type?: "conversa" | "quiz" | "pesquisa";
}

export function ChatAssistente({
    initialMessages = [],
    type = "conversa",
}: ChatAssistenteProps) {
    const [messages, setMessages] = useState<Message[]>(initialMessages);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [selectedChatModel, setSelectedChatModel] = useState<
        string | undefined
    >();

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
        setIsLoading(true);

        try {
            const response = await fetch(`/api/chatbot/${type}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: input,
                    history: messages.map((msg) => ({
                        role: msg.role,
                        content: msg.content,
                    })),
                }),
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();

            const aiResponse: Message = {
                id: (Date.now() + 1).toString(),
                content: data.response,
                role: "assistant",
            };

            setMessages((prev) => [...prev, aiResponse]);
        } catch (error) {
            console.error("Erro:", error);
            // Adiciona mensagem de erro ao chat
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                content: "Desculpe, ocorreu um erro ao processar sua mensagem.",
                role: "assistant",
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-4">
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
